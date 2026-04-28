import argparse
import json
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


def normalize_observation(observation: str) -> str:
    observation = str(observation).strip()
    if not observation.startswith('OBS '):
        observation = 'OBS ' + observation
    return observation


def load_checkpoint_metadata(checkpoint_path: str) -> dict:
    if not checkpoint_path:
        return {}
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint if isinstance(checkpoint, dict) else {}


def create_base_model(args, ntoken, special_tokens, model_runtime_config):
    return create_transformer(
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_name=args.modelname,
        vocab_size=ntoken,
        use_pretrained_weights=args.use_pretrained_text_model,
        model_runtime_config=model_runtime_config,
    )


def load_inference_model(args, tokenizer, ntoken, special_tokens, model_runtime_config, device):
    checkpoint_path = str(args.checkpoint_path or '').strip()
    if not checkpoint_path:
        model = create_base_model(args, ntoken, special_tokens, model_runtime_config)
        model.to(device)
        model.eval()
        return model

    checkpoint = load_checkpoint_metadata(checkpoint_path)
    base_model = None
    if checkpoint.get('peft_checkpoint'):
        base_model = create_base_model(args, ntoken, special_tokens, model_runtime_config)

    model, _, _, _, _ = load_model(
        checkpoint_path,
        'model',
        return_huggingface_model=True,
        epoch=args.resume_epoch,
        model=base_model,
    )
    model.to(device)
    model.eval()
    return model


def generate_completion(model, tokenizer, prompt, device, args):
    tokenized = tokenizer(prompt, return_tensors='pt').to(device)
    input_length = tokenized.input_ids.shape[-1]
    generation_kwargs = {
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask,
        'max_new_tokens': args.max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': args.do_sample,
        'repetition_penalty': args.repetition_penalty,
    }
    if args.do_sample:
        generation_kwargs['temperature'] = args.temperature
        if args.top_k > 0:
            generation_kwargs['top_k'] = args.top_k
        if args.top_p < 1.0:
            generation_kwargs['top_p'] = args.top_p

    with torch.no_grad():
        output = model.generate(**generation_kwargs)
    generated_ids = output[0, input_length:].detach().cpu().tolist()
    return decode_text_token_ids(tokenizer, generated_ids, preserve_whitespace=True)


def build_prompt(observation_text, history):
    return '\n'.join([observation_text, *history]).strip()


def run_stage1(args, model, tokenizer, device):
    observation = normalize_observation(args.observation)
    generated = generate_completion(model, tokenizer, observation, device, args)
    return {
        'stage': 'logic',
        'observation': observation,
        'history': [],
        'raw_generations': [{'step': 'stage1', 'prompt': observation, 'generated': generated}],
        'dsl': extract_dsl_text(generated) or '',
        'final_text': generated.strip(),
        'stopped_by': 'stage1_generation',
    }


def run_stage2(args, model, tokenizer, kg, device):
    observation = normalize_observation(args.observation)
    history = []
    raw_generations = []

    for step in range(1, args.max_action_steps + 1):
        prompt = build_prompt(observation, history)
        generated = generate_completion(model, tokenizer, prompt, device, args)
        raw_generations.append({'step': step, 'prompt': prompt, 'generated': generated})

        dsl_text = extract_dsl_text(generated)
        if dsl_text is not None:
            return {
                'stage': 'stage2_loop',
                'observation': observation,
                'history': history,
                'raw_generations': raw_generations,
                'dsl': dsl_text,
                'final_text': dsl_text,
                'stopped_by': 'dsl',
            }

        action_text = extract_action_text(generated)
        if action_text is None:
            return {
                'stage': 'stage2_loop',
                'observation': observation,
                'history': history,
                'raw_generations': raw_generations,
                'dsl': '',
                'final_text': generated.strip(),
                'stopped_by': 'unparseable_generation',
            }

        try:
            tool_output = run_action_tool_call(
                observation_text=observation,
                action_text=action_text,
                kg=kg,
                graph_split=args.graph_split,
            )
        except Exception as exc:
            history.append(action_text)
            raw_generations.append({'step': step, 'tool_error': str(exc)})
            return {
                'stage': 'stage2_loop',
                'observation': observation,
                'history': history,
                'raw_generations': raw_generations,
                'dsl': '',
                'final_text': '',
                'stopped_by': 'action_execution_error',
            }
        history.extend([action_text, tool_output['result_text']])

    prompt = build_prompt(observation, history)
    generated = generate_completion(model, tokenizer, prompt, device, args)
    raw_generations.append({'step': 'final', 'prompt': prompt, 'generated': generated})
    dsl_text = extract_dsl_text(generated) or ''
    return {
        'stage': 'stage2_loop',
        'observation': observation,
        'history': history,
        'raw_generations': raw_generations,
        'dsl': dsl_text,
        'final_text': dsl_text or generated.strip(),
        'stopped_by': 'max_action_steps',
    }


def print_result(result, print_raw=False, json_output=False):
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print('=== STAGE ===')
    print(result['stage'])
    print('=== OBSERVATION ===')
    print(result['observation'])
    if result.get('history'):
        print('=== ACTION_RESULT_HISTORY ===')
        print('\n'.join(result['history']))
    print('=== FINAL_TEXT ===')
    print(result.get('final_text', ''))
    if result.get('dsl'):
        print('=== DSL ===')
        print(result['dsl'])
    print('=== STOPPED_BY ===')
    print(result.get('stopped_by', ''))
    if print_raw:
        print('=== RAW_GENERATIONS ===')
        for item in result.get('raw_generations', []):
            print(f'# step={item.get("step")}')
            if 'prompt' in item:
                print('PROMPT:')
                print(item['prompt'])
            if 'generated' in item:
                print('GENERATED:')
                print(item['generated'])
            if 'tool_error' in item:
                print('TOOL_ERROR:')
                print(item['tool_error'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observation', required=True, help='Example: "OBS [A] [B]"')
    parser.add_argument('--stage', choices=['logic', 'stage1', 'stage2_loop'], default='stage2_loop')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--modelname', default='GPT2_6_act_nt')
    parser.add_argument('--config-model', default='configs/config-model.yml')
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--graph-split', default='train')
    parser.add_argument('--max-action-steps', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--repetition-penalty', type=float, default=1.05)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--use-pretrained-text-model', action='store_true')
    parser.add_argument(
        '--disable-text-extra-tokens',
        action='store_true',
        help='Use this if the checkpoint was trained with training.py --disable_text_extra_tokens.',
    )
    parser.add_argument('--device', default='')
    parser.add_argument('--print-raw', action='store_true')
    parser.add_argument('--json-output', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == 'stage1':
        args.stage = 'logic'
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config_model = load_yaml(args.config_model)
    model_runtime_config = resolve_model_runtime_config(args.modelname, config_model)
    text_extra_tokens = [] if args.disable_text_extra_tokens else get_text_extra_tokens(include_graph_tokens=False)
    tokenizer, ntoken = create_text_tokenizer(
        get_tokenizer_path(model_runtime_config),
        extra_tokens=text_extra_tokens,
        closed_text_tokens=None,
        trust_remote_code=bool(model_runtime_config.get('trust_remote_code', False)),
    )
    special_tokens = {
        'PAD': tokenizer.pad_token_id,
        'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        'END': tokenizer.eos_token_id,
    }
    model = load_inference_model(
        args=args,
        tokenizer=tokenizer,
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_runtime_config=model_runtime_config,
        device=device,
    )
    if args.stage == 'logic':
        result = run_stage1(args, model=model, tokenizer=tokenizer, device=device)
    else:
        kg = load_kg(args.dataname)
        result = run_stage2(args, model=model, tokenizer=tokenizer, kg=kg, device=device)
    print_result(result, print_raw=args.print_raw, json_output=args.json_output)


if __name__ == '__main__':
    main()
