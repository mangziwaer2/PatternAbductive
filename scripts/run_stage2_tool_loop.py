import argparse
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from model.tokenizer import create_text_tokenizer, decode_text_token_ids, get_text_extra_tokens
from model.transformer import create_transformer, get_tokenizer_path, resolve_model_runtime_config
from utils.load import load_kg, load_model, load_yaml
from utils.tool_loop import extract_action_text, extract_dsl_text, run_action_tool_call


def resolve_checkpoint_path(args):
    if args.checkpoint_path:
        return args.checkpoint_path
    if args.resume_epoch <= 0:
        return ''
    candidates = [
        os.path.join(
            args.checkpoint_root,
            args.modelname,
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}-text2text.pth',
        ),
        os.path.join(
            args.checkpoint_root,
            args.modelname,
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}-conditioned.pth',
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_stage2_model(args, tokenizer, ntoken, device):
    config_model = load_yaml(args.config_model)
    model_runtime_config = resolve_model_runtime_config(args.modelname, config_model)
    checkpoint_path = resolve_checkpoint_path(args)
    if checkpoint_path:
        model, _, _, _, _ = load_model(
            checkpoint_path,
            'model',
            return_huggingface_model=True,
            epoch=args.resume_epoch,
        )
        model.to(device)
        model.eval()
        return model

    special_tokens = {
        'PAD': tokenizer.pad_token_id,
        'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        'END': tokenizer.eos_token_id,
    }
    model = create_transformer(
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_name=args.modelname,
        vocab_size=ntoken,
        use_pretrained_weights=args.use_pretrained_text_model,
        model_runtime_config=model_runtime_config,
    ).to(device)
    model.eval()
    return model


def generate_completion(model, tokenizer, prompt, device, max_new_tokens, top_k=0, do_sample=False):
    tokenized = tokenizer(prompt, return_tensors='pt').to(device)
    input_length = tokenized.input_ids.shape[-1]
    with torch.no_grad():
        generation_kwargs = {
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask,
            'max_length': input_length + max_new_tokens,
            'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'do_sample': do_sample,
        }
        if do_sample and top_k > 0:
            generation_kwargs['top_k'] = top_k
        output = model.generate(**generation_kwargs)
    generated_ids = output[0, input_length:].tolist()
    return decode_text_token_ids(tokenizer, generated_ids, preserve_whitespace=True)


def build_next_prompt(observation_text, history):
    return '\n'.join([observation_text, *history])


def run_loop(args, model, tokenizer, kg, device):
    observation = args.observation.strip()
    if not observation.startswith('OBS '):
        observation = 'OBS ' + observation

    history = []
    raw_generations = []

    for step in range(1, args.max_action_steps + 1):
        prompt = build_next_prompt(observation, history)
        generated = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            do_sample=args.do_sample,
        )
        raw_generations.append({'step': step, 'prompt': prompt, 'generated': generated})

        dsl_text = extract_dsl_text(generated)
        if dsl_text is not None:
            return {
                'observation': observation,
                'history': history,
                'dsl': dsl_text,
                'raw_generations': raw_generations,
                'stopped_by': 'dsl',
            }

        action_text = extract_action_text(generated)
        if action_text is None:
            return {
                'observation': observation,
                'history': history,
                'dsl': '',
                'raw_generations': raw_generations,
                'stopped_by': 'unparseable_generation',
            }

        tool_output = run_action_tool_call(
            observation_text=observation,
            action_text=action_text,
            kg=kg,
            graph_split=args.graph_split,
        )
        history.extend([action_text, tool_output['result_text']])

    final_prompt = build_next_prompt(observation, history)
    generated = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=final_prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        do_sample=args.do_sample,
    )
    raw_generations.append({'step': 'final', 'prompt': final_prompt, 'generated': generated})
    return {
        'observation': observation,
        'history': history,
        'dsl': extract_dsl_text(generated) or generated.strip(),
        'raw_generations': raw_generations,
        'stopped_by': 'max_action_steps',
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observation', required=True)
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--modelname', default='GPT2_6_act_nt')
    parser.add_argument('--config-model', default='configs/config-model.yml')
    parser.add_argument('--config-train', default='configs/config-train.yml')
    parser.add_argument('--checkpoint-root', default='./ckpt/')
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--scale', default='default')
    parser.add_argument('--max-answer-size', type=int, default=8)
    parser.add_argument('--graph-split', default='train')
    parser.add_argument('--max-action-steps', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--use-pretrained-text-model', action='store_true')
    parser.add_argument('--device', default='')
    parser.add_argument('--print-raw', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    kg = load_kg(args.dataname)
    config_model = load_yaml(args.config_model)
    model_runtime_config = resolve_model_runtime_config(args.modelname, config_model)
    tokenizer, ntoken = create_text_tokenizer(
        get_tokenizer_path(model_runtime_config),
        extra_tokens=get_text_extra_tokens(include_graph_tokens=False),
        closed_text_tokens=None,
        trust_remote_code=bool(model_runtime_config.get('trust_remote_code', False)),
    )
    model = load_stage2_model(args, tokenizer, ntoken, device)
    result = run_loop(args, model=model, tokenizer=tokenizer, kg=kg, device=device)

    print('=== OBSERVATION ===')
    print(result['observation'])
    print('=== HISTORY ===')
    print('\n'.join(result['history']))
    print('=== DSL ===')
    print(result['dsl'])
    print('=== STOPPED_BY ===')
    print(result['stopped_by'])
    if args.print_raw:
        print('=== RAW_GENERATIONS ===')
        for item in result['raw_generations']:
            print(f'# step={item["step"]}')
            print(item['generated'])


if __name__ == '__main__':
    main()
