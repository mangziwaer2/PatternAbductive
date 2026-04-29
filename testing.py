import argparse
import os
import random
from pathlib import Path

import torch

from model.tokenizer import create_text_tokenizer, get_text_extra_tokens
from model.transformer import (
    create_transformer,
    get_tokenizer_path,
    resolve_model_runtime_config,
)
from utils.generation_control import stream_generate_until_tags
from utils.load import load_kg, load_model, load_yaml
from utils.logic_dsl import DSL_END_TAG
from utils.tool_loop import extract_action_text, extract_dsl_text, run_action_tool_call


PIPELINE_TAG = 'text2text'


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def normalize_observation(obs: str) -> str:
    obs = str(obs or '').strip()
    if not obs:
        raise ValueError('Empty observation. Pass --obs or --obs-file.')
    return obs if obs.startswith('OBS ') else f'OBS {obs}'


def read_observation(args) -> str:
    if args.obs_file:
        with open(args.obs_file, 'r', encoding='utf-8') as input_file:
            return normalize_observation(input_file.read())
    if args.obs:
        return normalize_observation(args.obs)
    return normalize_observation(input('OBS> '))


def default_checkpoint_path(args, contents: str) -> str:
    if args.resume_epoch <= 0:
        return ''
    middle = f'{args.resume_epoch}-rl' if contents == 'rlmodel' else str(args.resume_epoch)
    return os.path.join(
        args.checkpoint_root,
        args.modelname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}-{middle}-{PIPELINE_TAG}.pth',
    )


def find_single_checkpoint_file(path: str) -> str:
    path_obj = Path(path)
    candidates = sorted(path_obj.glob(f'*{PIPELINE_TAG}.pth'))
    if not candidates:
        candidates = sorted(path_obj.glob('*.pth'))
    if len(candidates) == 1:
        return str(candidates[0])
    return ''


def has_adapter_config(path: str) -> bool:
    return bool(path) and os.path.exists(os.path.join(str(path), 'adapter_config.json'))


def checkpoint_has_nearby_adapter(checkpoint_path: str) -> bool:
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if path.is_dir():
        if has_adapter_config(str(path)):
            return True
        checkpoint_file = find_single_checkpoint_file(str(path))
        if checkpoint_file:
            return checkpoint_has_nearby_adapter(checkpoint_file)
        return any(has_adapter_config(str(child)) for child in path.iterdir() if child.is_dir())

    checkpoint_dir = path.parent
    candidates = [
        f'{checkpoint_path}.adapter',
        str(checkpoint_dir / f'{path.name}.adapter'),
        str(checkpoint_dir / f'{path.stem}.adapter'),
        str(checkpoint_dir),
    ]
    if path.name.endswith('-text2text.pth'):
        candidates.append(str(checkpoint_dir / path.name[:-len('-text2text.pth')]))
    if path.suffix == '.pth':
        candidates.append(str(checkpoint_dir / path.stem))
    return any(has_adapter_config(candidate) for candidate in candidates)


def create_base_model(args, ntoken, special_tokens, model_runtime_config, device):
    model = create_transformer(
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_name=args.modelname,
        vocab_size=ntoken,
        use_pretrained_weights=args.use_pretrained_text_model,
        model_runtime_config=model_runtime_config,
    )
    return model.to(device)


def resolve_checkpoint_contents(args, checkpoint_path: str) -> str:
    if args.checkpoint_contents != 'auto':
        return args.checkpoint_contents
    name = os.path.basename(str(checkpoint_path)).lower()
    if '-rl-' in name or '-optimize-' in name:
        return 'rlmodel'
    return 'model'


def load_inference_model(args, ntoken, special_tokens, model_runtime_config, device):
    checkpoint_path = str(args.checkpoint_path or '').strip()
    contents = resolve_checkpoint_contents(args, checkpoint_path)
    if not checkpoint_path:
        checkpoint_path = default_checkpoint_path(args, contents)

    if not checkpoint_path:
        print('# No checkpoint supplied; creating a fresh model.')
        return create_base_model(args, ntoken, special_tokens, model_runtime_config, device)

    needs_base_model = args.use_peft or checkpoint_has_nearby_adapter(checkpoint_path)
    base_model = (
        create_base_model(args, ntoken, special_tokens, model_runtime_config, device)
        if needs_base_model else None
    )

    try:
        model, _, _, _, _ = load_model(
            checkpoint_path,
            contents,
            return_huggingface_model=True,
            model=base_model,
        )
    except ValueError as exc:
        if base_model is not None or 'PEFT/LoRA' not in str(exc):
            raise
        base_model = create_base_model(args, ntoken, special_tokens, model_runtime_config, device)
        model, _, _, _, _ = load_model(
            checkpoint_path,
            contents,
            return_huggingface_model=True,
            model=base_model,
        )

    model.model_name = args.modelname
    return model.to(device)


def generate_segment(model, tokenizer, prompt, device, args, stop_strings=None):
    return stream_generate_until_tags(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_strings=stop_strings,
    )


def run_logic_inference(model, tokenizer, observation, device, args):
    segment = generate_segment(
        model=model,
        tokenizer=tokenizer,
        prompt=observation,
        device=device,
        args=args,
        stop_strings=[DSL_END_TAG],
    )
    output = segment['completion'].strip()
    print('===== INPUT =====')
    print(observation)
    print('===== OUTPUT =====')
    print(output)
    dsl = extract_dsl_text(output)
    if dsl is not None:
        print('===== DSL =====')
        print(dsl)
    return output


def run_stage2_inference(model, tokenizer, kg, observation, device, args):
    history = []
    transcript = [observation]

    print('===== INPUT =====')
    print(observation)

    for step in range(1, args.max_action_steps + 1):
        prompt = '\n'.join([observation, *history]).strip()
        segment = generate_segment(model, tokenizer, prompt, device, args)
        generated = segment['completion'].strip()

        action_text = extract_action_text(generated)
        if action_text is not None:
            print(f'===== ACTION {step} =====')
            print(action_text)
            tool_output = run_action_tool_call(
                observation_text=observation,
                action_text=action_text,
                kg=kg,
                graph_split=args.search_split,
            )
            result_text = tool_output['result_text']
            print(f'===== RESULT {step} =====')
            print(result_text)
            history.extend([action_text, result_text])
            transcript.extend([action_text, result_text])
            continue

        dsl = extract_dsl_text(generated)
        if dsl is not None:
            print('===== DSL =====')
            print(dsl)
            transcript.append(generated)
            return '\n'.join(transcript)

        print('===== UNPARSEABLE GENERATION =====')
        print(generated)
        transcript.append(generated)
        return '\n'.join(transcript)

    prompt = '\n'.join([observation, *history]).strip()
    segment = generate_segment(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        args=args,
        stop_strings=[DSL_END_TAG],
    )
    generated = segment['completion'].strip()
    dsl = extract_dsl_text(generated)
    print('===== FINAL GENERATION =====')
    print(generated)
    if dsl is not None:
        print('===== DSL =====')
        print(dsl)
    transcript.append(generated)
    return '\n'.join(transcript)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default="Qwen2.5-0.5B")
    parser.add_argument('--config-model', default='configs/config-model.yml')
    parser.add_argument('--dataname', default='DBpedia50')
    parser.add_argument('--scale', default='default')
    parser.add_argument('--max-answer-size', type=int, default=8)
    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument('--checkpoint-path', dest='checkpoint_path', default=r'E:\project\LLM\PatternAbductive\ckpt\Qwen2.5-0.5B')
    parser.add_argument('--checkpoint_contents', choices=['auto', 'model', 'rlmodel'], default='auto')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--stage', choices=['logic', 'stage2'], default='stage2')
    parser.add_argument('--obs', default='')
    parser.add_argument('--obs-file', default='')
    parser.add_argument('--search_split', default='train')
    parser.add_argument('--max_action_steps', type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--do_sample', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--use_pretrained_text_model', action='store_true')
    parser.add_argument('--disable_text_extra_tokens', action='store_true')
    parser.add_argument('--use_peft', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_runtime_config=model_runtime_config,
        device=device,
    )
    model.eval()

    observation = read_observation(args)
    if args.stage == 'logic':
        run_logic_inference(model, tokenizer, observation, device, args)
        return

    kg = load_kg(args.dataname)
    run_stage2_inference(model, tokenizer, kg, observation, device, args)


if __name__ == '__main__':
    main()
