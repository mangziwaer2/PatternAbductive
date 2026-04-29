import argparse
import csv
import datetime
import json
import logging
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import pandas as pd
import torch
from tqdm import tqdm

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

from model.tokenizer import (
    create_text_tokenizer,
    decode_text_token_ids,
    extract_text_sample_to_device,
    get_text_extra_tokens,
)
from model.transformer import (
    create_transformer,
    get_tokenizer_path,
    resolve_model_runtime_config,
)
from utils.dataloader import (
    new_create_dataloader,
    new_create_dataset,
    record_has_current_stage2_trace,
)
from utils.load import load_kg, load_model, load_yaml, resolve_sampled_dataset_path
from utils.rl_rewards import score_rollout_trajectory
from utils.tool_loop import extract_action_text, extract_dsl_text, run_action_tool_call
from utils.generation_control import stream_generate_until_tags


PIPELINE_TAG = 'text2text'
LEGACY_PIPELINE_TAG = 'conditioned'
DEFAULT_SOURCE_TEXT_FIELD = 'observation_text'
DEFAULT_TARGET_TEXT_FIELD = 'hypothesis_text'
DEFAULT_CONDITION_TEXT_FIELD = 'condition_text'


def run_command(command):
    try:
        return subprocess.check_output(
            command,
            cwd=os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ''


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_experiment_name(args):
    if args.experiment_name:
        return args.experiment_name
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'{timestamp}-{args.dataname}-{args.modelname}-{PIPELINE_TAG}'


def get_runtime_metadata(device):
    metadata = {
        'device': str(device),
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(current_device)
        metadata.update({
            'gpu_name': torch.cuda.get_device_name(current_device),
            'gpu_total_memory_mb': round(device_props.total_memory / (1024 ** 2), 2),
        })
    return metadata


def initialize_csv(csv_path, fieldnames):
    pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(csv_path, fieldnames, row):
    with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


def append_text_log(log_path, message):
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message + '\n')


def emit_text_log(message, log_path=None, also_print=False):
    if log_path is not None:
        append_text_log(log_path, message)
    if also_print:
        print(message, flush=True)


def format_seconds(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def should_emit_periodic_log(step, total_steps, every):
    if total_steps <= 0:
        return False
    if step == 1 or step == total_steps:
        return True
    return every > 0 and step % every == 0


def average_or_none(values):
    return (sum(values) / len(values)) if values else None


def split_csv_arg(value):
    return [
        item.strip()
        for item in str(value or '').split(',')
        if item.strip()
    ]


def infer_lora_target_modules(model):
    model_type = str(getattr(getattr(model, 'config', None), 'model_type', '') or '').lower()
    if model_type == 'gpt2':
        return ['c_attn', 'c_proj', 'c_fc']

    known_targets = [
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
        'c_attn',
        'c_proj',
        'c_fc',
    ]
    module_names = {
        name.rsplit('.', 1)[-1]
        for name, _ in model.named_modules()
    }
    return [
        target
        for target in known_targets
        if target in module_names
    ]


def infer_lora_modules_to_save(model):
    module_names = {
        name.rsplit('.', 1)[-1]
        for name, _ in model.named_modules()
    }
    preferred = ['embed_tokens', 'lm_head', 'wte']
    return [
        module_name
        for module_name in preferred
        if module_name in module_names
    ]


def resolve_lora_modules_to_save(model, raw_value):
    value = str(raw_value or 'auto').strip()
    if value.lower() in {'', 'auto'}:
        return infer_lora_modules_to_save(model)
    if value.lower() in {'none', 'false', '0'}:
        return []
    return split_csv_arg(value)


def is_peft_model(model):
    return hasattr(model, 'peft_config') or model.__class__.__name__.lower().startswith('peft')


def apply_lora_if_requested(model, args):
    if not args.use_peft:
        return model, False
    if is_peft_model(model):
        print('# LoRA/PEFT already present in loaded model.')
        return model, False

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError('peft is required for --use_peft. Install it with `pip install peft`.') from exc

    target_modules = split_csv_arg(args.lora_target_modules)
    if not target_modules:
        target_modules = infer_lora_target_modules(model)
    if not target_modules:
        raise ValueError(
            'Could not infer LoRA target modules. Set --lora_target_modules explicitly, '
            'for example: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
        )
    modules_to_save = resolve_lora_modules_to_save(model, args.lora_modules_to_save)

    model_type = str(getattr(getattr(model, 'config', None), 'model_type', '') or '').lower()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type='CAUSAL_LM',
        target_modules=target_modules,
        fan_in_fan_out=(model_type == 'gpt2'),
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, peft_config=lora_config)
    print(f'# Enabled LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}')
    print(f'# LoRA target modules: {target_modules}')
    print(f'# LoRA modules_to_save: {modules_to_save or []}')
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    return model, True


def create_optimizer_and_scheduler(model, config_train):
    trainable_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError('No trainable parameters found. Check --use_peft / LoRA configuration.')
    optimizer = torch.optim.Adam(trainable_params, lr=float(config_train['lr']))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config_train['warm_up'],
    )
    return optimizer, scheduler


def collect_dataset_sizes(dataset_dict):
    return {
        split: int(dataset.shape[0])
        for split, dataset in dataset_dict.items()
    }


def _read_first_jsonl_record(path):
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def training_data_needs_kg(args, splits):
    if args.force_load_kg or args.mode != 'training':
        return True

    for split in splits:
        try:
            data_path = resolve_sampled_dataset_path(args.data_root, args.dataname, split)
            first_record = _read_first_jsonl_record(data_path)
        except FileNotFoundError:
            raise
        except Exception:
            return True

        if 'logic_dsl' not in first_record:
            return True
        if args.train_stage == 'stage2' and not record_has_current_stage2_trace(first_record):
            return True

    return False


def sanitize_filename_component(value):
    text = str(value)
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    return text


def prepare_experiment_record(args, dataset_dict, config_train, config_dataloader, device):
    experiment_name = build_experiment_name(args)
    experiment_dir = os.path.join(args.experiment_root, experiment_name)
    pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        'experiment_dir': experiment_dir,
        'config_path': os.path.join(experiment_dir, 'config.json'),
        'loss_csv_path': os.path.join(experiment_dir, 'train_valid_loss.csv'),
        'comparison_log_path': os.path.join(experiment_dir, 'prediction_vs_groundtruth.log'),
        'summary_path': os.path.join(experiment_dir, 'summary.md'),
        'run_log_path': os.path.join(experiment_dir, 'run.log'),
    }

    for clear_path in [paths['comparison_log_path'], paths['run_log_path']]:
        if os.path.exists(clear_path):
            os.remove(clear_path)

    metadata = {
        'experiment_name': experiment_name,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'git': {
            'commit': run_command(['git', 'rev-parse', 'HEAD']),
            'branch': run_command(['git', 'branch', '--show-current']),
        },
        'runtime': get_runtime_metadata(device),
        'data': {
            'dataname': args.dataname,
            'data_root': args.data_root,
            'splits_used': list(dataset_dict.keys()),
            'split_sizes': collect_dataset_sizes(dataset_dict),
            'pipeline': f'train_stage={args.train_stage}',
        },
        'args': vars(args),
        'config_train': config_train,
        'config_dataloader': config_dataloader,
        'artifacts': {
            'loss_csv': paths['loss_csv_path'],
            'comparison_log': paths['comparison_log_path'],
            'summary': paths['summary_path'],
            'run_log': paths['run_log_path'],
        },
    }

    with open(paths['config_path'], 'w', encoding='utf-8') as config_file:
        json.dump(metadata, config_file, ensure_ascii=False, indent=2)

    initialize_csv(paths['loss_csv_path'], ['epoch', 'train_loss', 'valid_loss'])
    append_text_log(paths['comparison_log_path'], '# Prediction vs Groundtruth')
    append_text_log(
        paths['comparison_log_path'],
        f'# comparison_samples={args.comparison_samples}, comparison_frequency={args.comparison_frequency}, '
        f'comparison_random={args.comparison_random}',
    )
    return {
        'name': experiment_name,
        'paths': paths,
        'metadata': metadata,
    }


def prepare_rl_experiment_record(args, dataset_dict, device):
    experiment_name = build_experiment_name(args)
    experiment_dir = os.path.join(args.rl_experiment_root, experiment_name)
    pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        'experiment_dir': experiment_dir,
        'config_path': os.path.join(experiment_dir, 'config.json'),
        'comparison_log_path': os.path.join(experiment_dir, 'prediction_vs_groundtruth.log'),
        'summary_path': os.path.join(experiment_dir, 'summary.md'),
        'run_log_path': os.path.join(experiment_dir, 'run.log'),
    }

    for clear_path in [paths['comparison_log_path'], paths['run_log_path']]:
        if os.path.exists(clear_path):
            os.remove(clear_path)

    metadata = {
        'experiment_name': experiment_name,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'git': {
            'commit': run_command(['git', 'rev-parse', 'HEAD']),
            'branch': run_command(['git', 'branch', '--show-current']),
        },
        'runtime': get_runtime_metadata(device),
        'data': {
            'dataname': args.dataname,
            'data_root': args.data_root,
            'splits_used': list(dataset_dict.keys()),
            'split_sizes': collect_dataset_sizes(dataset_dict),
            'pipeline': f'train_stage={args.train_stage}',
        },
        'args': vars(args),
        'artifacts': {
            'comparison_log': paths['comparison_log_path'],
            'summary': paths['summary_path'],
            'run_log': paths['run_log_path'],
        },
    }

    with open(paths['config_path'], 'w', encoding='utf-8') as config_file:
        json.dump(metadata, config_file, ensure_ascii=False, indent=2)

    append_text_log(paths['comparison_log_path'], '# Prediction vs Groundtruth')
    append_text_log(
        paths['comparison_log_path'],
        f'# comparison_samples={args.comparison_samples}, comparison_frequency={args.comparison_frequency}, '
        f'comparison_random={args.comparison_random}',
    )
    return {
        'name': experiment_name,
        'paths': paths,
        'metadata': metadata,
    }


def write_experiment_summary(experiment_record, loss_log):
    train_history = sorted(loss_log['train'].items())
    valid_history = sorted(loss_log['valid'].items())

    lines = [
        f'# {experiment_record["name"]}',
        '',
        f'- created_at: {experiment_record["metadata"]["created_at"]}',
        f'- git_commit: {experiment_record["metadata"]["git"]["commit"] or "unknown"}',
        f'- git_branch: {experiment_record["metadata"]["git"]["branch"] or "unknown"}',
        f'- device: {experiment_record["metadata"]["runtime"]["device"]}',
    ]

    gpu_name = experiment_record['metadata']['runtime'].get('gpu_name')
    if gpu_name:
        lines.append(f'- gpu: {gpu_name}')

    lines.extend([
        f'- splits_used: {", ".join(experiment_record["metadata"]["data"]["splits_used"])}',
        f'- split_sizes: {json.dumps(experiment_record["metadata"]["data"]["split_sizes"], ensure_ascii=False)}',
        f'- pipeline: {experiment_record["metadata"]["data"]["pipeline"]}',
        '',
        '## Loss Summary',
    ])

    if train_history:
        first_epoch, first_train = train_history[0]
        last_epoch, last_train = train_history[-1]
        lines.append(f'- train_loss: epoch {first_epoch} {first_train:.6f} -> epoch {last_epoch} {last_train:.6f}')
    else:
        lines.append('- train_loss: unavailable')

    if valid_history:
        first_valid_epoch, first_valid = valid_history[0]
        last_valid_epoch, last_valid = valid_history[-1]
        best_valid_epoch, best_valid = min(valid_history, key=lambda item: item[1])
        lines.append(f'- valid_loss: epoch {first_valid_epoch} {first_valid:.6f} -> epoch {last_valid_epoch} {last_valid:.6f}')
        lines.append(f'- best_valid_loss: epoch {best_valid_epoch} {best_valid:.6f}')
    else:
        lines.append('- valid_loss: unavailable')

    lines.extend([
        '',
        '## Artifacts',
        f'- config: {experiment_record["paths"]["config_path"]}',
        f'- loss_csv: {experiment_record["paths"]["loss_csv_path"]}',
        f'- comparison_log: {experiment_record["paths"]["comparison_log_path"]}',
        f'- run_log: {experiment_record["paths"]["run_log_path"]}',
    ])

    with open(experiment_record['paths']['summary_path'], 'w', encoding='utf-8') as summary_file:
        summary_file.write('\n'.join(lines) + '\n')


def write_rl_experiment_summary(experiment_record, trainer_result):
    metrics = trainer_result.metrics if trainer_result is not None else {}
    lines = [
        f'# {experiment_record["name"]}',
        '',
        f'- created_at: {experiment_record["metadata"]["created_at"]}',
        f'- git_commit: {experiment_record["metadata"]["git"]["commit"] or "unknown"}',
        f'- git_branch: {experiment_record["metadata"]["git"]["branch"] or "unknown"}',
        f'- device: {experiment_record["metadata"]["runtime"]["device"]}',
        f'- data_root: {experiment_record["metadata"]["data"]["data_root"]}',
        f'- splits_used: {", ".join(experiment_record["metadata"]["data"]["splits_used"])}',
        f'- pipeline: {experiment_record["metadata"]["data"]["pipeline"]}',
        '',
        '## RL Metrics',
    ]

    if metrics:
        for key in sorted(metrics):
            lines.append(f'- {key}: {metrics[key]}')
    else:
        lines.append('- metrics: unavailable')

    lines.extend([
        '',
        '## Artifacts',
        f'- config: {experiment_record["paths"]["config_path"]}',
        f'- comparison_log: {experiment_record["paths"]["comparison_log_path"]}',
        f'- run_log: {experiment_record["paths"]["run_log_path"]}',
    ])

    with open(experiment_record['paths']['summary_path'], 'w', encoding='utf-8') as summary_file:
        summary_file.write('\n'.join(lines) + '\n')


def wrap_single_sample(sample):
    return {key: [value] for key, value in sample.items()}


def select_sample_indices(dataset_size, num_samples, randomize=False, seed=0):
    if dataset_size <= 0 or num_samples <= 0:
        return []
    if num_samples >= dataset_size:
        return list(range(dataset_size))
    if randomize:
        rng = random.Random(int(seed))
        return sorted(rng.sample(range(dataset_size), num_samples))
    if num_samples == 1:
        return [0]

    stride = (dataset_size - 1) / (num_samples - 1)
    indices = []
    for idx in range(num_samples):
        candidate = min(int(round(idx * stride)), dataset_size - 1)
        if not indices or candidate != indices[-1]:
            indices.append(candidate)
    return indices


def format_logged_condition(condition_value):
    if condition_value is None:
        return ''
    if isinstance(condition_value, float) and pd.isna(condition_value):
        return ''
    return str(condition_value)


def build_logged_input(source_value, condition_value):
    condition_value = format_logged_condition(condition_value).strip()
    parts = [source_value]
    if condition_value:
        parts.extend(['SEP', condition_value])
    return ' '.join(parts)


def run_generation(model, input_ids, attention_mask, tokenizer, max_length, top_k=0, do_sample=True):
    max_new_tokens = max(1, int(max_length) - int(input_ids.shape[1]))
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=1.0,
        do_sample=do_sample,
    )
    if do_sample and top_k > 0:
        generation_kwargs['top_k'] = top_k
    return model.generate(**generation_kwargs)


@torch.no_grad()
def log_prediction_comparisons(
        args,
        dataset_dict,
        model,
        tokenizer,
        src_len,
        tgt_len,
        device,
        accelerator,
        log_path,
        stage_label):
    if args.comparison_samples <= 0:
        return

    model_for_generation = accelerator.unwrap_model(model) if accelerator is not None else model
    model_for_generation.eval()
    emit_text_log('', log_path, also_print=args.comparison_console)
    emit_text_log(f'===== {stage_label} =====', log_path, also_print=args.comparison_console)

    for split in ['train', 'valid']:
        if split not in dataset_dict:
            continue

        dataset = dataset_dict[split]
        seed_offset = sum(ord(char) for char in f'{stage_label}:{split}')
        indices = select_sample_indices(
            len(dataset),
            args.comparison_samples,
            randomize=args.comparison_random,
            seed=args.seed + seed_offset,
        )
        emit_text_log(f'[{split}] logged_indices={indices}', log_path, also_print=args.comparison_console)

        for sample_index in indices:
            sample = wrap_single_sample(dataset[sample_index])
            source, target, pattern_id, input_ids, attention_mask, _, source_attention_mask, condition = \
                extract_sample_batch(
                    args=args,
                    device=device,
                    sample=sample,
                    tokenizer=tokenizer,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    is_gen=True,
                )

            pred = run_generation(
                model=model_for_generation,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                max_length=input_ids.shape[1] + tgt_len,
                top_k=0,
                do_sample=False,
            )
            mask_source(device, source_attention_mask, pred, tokenizer)
            prediction = decode_text_token_ids(tokenizer, pred[0].tolist()).strip()

            condition_value = condition[0] if condition else ''

            emit_text_log(f'[{split}] idx={sample_index} pattern_id={pattern_id[0]}', log_path, also_print=args.comparison_console)
            emit_text_log(
                f'[{split}] INPUT  : {build_logged_input(source[0], condition_value)}',
                log_path,
                also_print=args.comparison_console,
            )
            emit_text_log(f'[{split}] TARGET : {target[0]}', log_path, also_print=args.comparison_console)
            emit_text_log(f'[{split}] PRED   : {prediction}', log_path, also_print=args.comparison_console)
            emit_text_log('', log_path, also_print=args.comparison_console)


def _extract_dsl_target_text(target: str) -> str:
    text = str(target or '').strip()
    tagged_dsl = extract_dsl_text(text)
    if tagged_dsl is not None:
        return tagged_dsl
    if ' DSL ' in text:
        return text.split(' DSL ', 1)[1].strip()
    if text.startswith('DSL '):
        return text[len('DSL '):].strip()
    return text


def _rollout_records_from_dataset(dataset):
    records = []
    for row in dataset:
        observation = str(row.get('observation_text') or row.get('source') or '').strip()
        if observation and not observation.startswith('OBS '):
            observation = 'OBS ' + observation
        target = _extract_dsl_target_text(row.get('logic_dsl') or row.get('target') or '')
        if observation:
            records.append({
                'observation_text': observation,
                'logic_dsl': target,
            })
    return records


def _build_rollout_prompt(observation_text, history):
    return '\n'.join([str(observation_text).strip(), *history]).strip()


@torch.no_grad()
def _stream_rollout_segment(model, tokenizer, prompt, device, args):
    return stream_generate_until_tags(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.rl_max_completion_length,
        do_sample=True,
        temperature=args.rl_temperature,
        top_k=args.rl_top_k,
        top_p=args.rl_top_p,
    )


def rollout_once(model, tokenizer, kg, record, device, args):
    observation = str(record['observation_text']).strip()
    if not observation.startswith('OBS '):
        observation = 'OBS ' + observation

    history = []
    segments = []
    raw_generations = []

    for step in range(1, args.rl_max_action_steps + 1):
        prompt = _build_rollout_prompt(observation, history)
        segment = _stream_rollout_segment(model, tokenizer, prompt, device, args)
        segments.append(segment)
        generated = segment['completion']
        raw_generations.append({'step': step, 'prompt': prompt, 'generated': generated})

        action_text = extract_action_text(generated)
        if action_text is not None:
            try:
                tool_output = run_action_tool_call(
                    observation_text=observation,
                    action_text=action_text,
                    kg=kg,
                    graph_split=args.rl_search_split,
                )
            except Exception as exc:
                history.append(action_text)
                return {
                    'observation': observation,
                    'history': history,
                    'dsl': '',
                    'raw_generations': [*raw_generations, {'step': step, 'tool_error': str(exc)}],
                    'segments': segments,
                    'stopped_by': 'action_execution_error',
                }
            history.extend([action_text, tool_output['result_text']])
            continue

        dsl_text = extract_dsl_text(generated)
        if dsl_text is not None:
            return {
                'observation': observation,
                'history': history,
                'dsl': dsl_text,
                'raw_generations': raw_generations,
                'segments': segments,
                'stopped_by': 'dsl',
            }

        return {
            'observation': observation,
            'history': history,
            'dsl': '',
            'raw_generations': raw_generations,
            'segments': segments,
            'stopped_by': 'unparseable_generation',
        }

    prompt = _build_rollout_prompt(observation, history)
    segment = _stream_rollout_segment(model, tokenizer, prompt, device, args)
    segments.append(segment)
    generated = segment['completion']
    raw_generations.append({'step': 'final', 'prompt': prompt, 'generated': generated})
    return {
        'observation': observation,
        'history': history,
        'dsl': extract_dsl_text(generated) or generated.strip(),
        'raw_generations': raw_generations,
        'segments': segments,
        'stopped_by': 'max_action_steps',
    }


def _segment_ce_loss(model, tokenizer, segment, device):
    generated_ids = [
        token_id
        for token_id in segment['generated_ids']
        if token_id != tokenizer.pad_token_id
    ]
    if not generated_ids:
        return None

    prompt_ids = tokenizer(segment['prompt'], return_tensors='pt').input_ids.to(device)
    generated = torch.tensor([generated_ids], dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, generated], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[-1]] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def _rollout_policy_loss(model, tokenizer, rollout, advantage, device):
    losses = []
    for segment in rollout.get('segments', []):
        loss = _segment_ce_loss(model, tokenizer, segment, device)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return None
    ce_loss = torch.stack(losses).mean()
    return ce_loss * float(advantage)


def optimize_rollout_policy(args, dataset, model, tokenizer, graph_samplers, kg, experiment_record=None):
    if kg is None or graph_samplers is None:
        raise RuntimeError('Rollout RL requires KG to execute ACTION calls.')

    records = _rollout_records_from_dataset(dataset)
    if not records:
        raise RuntimeError('No rollout records available for RL.')

    max_steps = args.rl_max_steps if args.rl_max_steps > 0 else max(1, int(args.rl_epochs) * len(records))
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError('No trainable parameters found for rollout RL.')
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.rl_lr))
    device = next(model.parameters()).device
    output_dir = (
        experiment_record['paths']['experiment_dir']
        if experiment_record is not None
        else os.path.join(args.rl_experiment_root, build_experiment_name(args))
    )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = (
        experiment_record['paths']['run_log_path']
        if experiment_record is not None
        else os.path.join(output_dir, 'rollout_rl.jsonl')
    )

    baseline = 0.0
    metrics = {
        'max_steps': max_steps,
        'records': len(records),
        'last_reward': 0.0,
        'last_loss': 0.0,
        'last_jaccard': 0.0,
        'last_num_actions': 0,
    }

    model.train()
    for step in range(1, max_steps + 1):
        record = random.choice(records)
        model.eval()
        rollout = rollout_once(model, tokenizer, kg, record, device, args)
        score = score_rollout_trajectory(
            rollout=rollout,
            target=record.get('logic_dsl', ''),
            observation_text=record['observation_text'],
            kg=kg,
            graph_samplers=graph_samplers,
            graph_split=args.rl_search_split,
        )
        reward = float(score['stage3_reward'])
        advantage = max(min(reward - baseline, args.rl_advantage_clip), -args.rl_advantage_clip)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _rollout_policy_loss(model, tokenizer, rollout, advantage, device)
        loss_value = 0.0
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.rl_grad_clip)
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        baseline = args.rl_baseline_momentum * baseline + (1.0 - args.rl_baseline_momentum) * reward
        row = {
            'step': step,
            'reward': reward,
            'baseline': baseline,
            'advantage': advantage,
            'loss': loss_value,
            'stopped_by': rollout.get('stopped_by', ''),
            'num_actions': score.get('num_actions', 0),
            'action_parse_rate': score.get('action_parse_rate', 0.0),
            'action_execution_rate': score.get('action_execution_rate', 0.0),
            'jaccard': score.get('jaccard', 0.0),
            'answer_recall': score.get('answer_recall', 0.0),
            'answer_precision': score.get('answer_precision', 0.0),
            'observation_text': record['observation_text'],
            'pred_dsl': rollout.get('dsl', ''),
        }
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(row, ensure_ascii=False) + '\n')

        metrics.update({
            'last_reward': reward,
            'last_loss': loss_value,
            'last_jaccard': row['jaccard'],
            'last_num_actions': row['num_actions'],
            'baseline': baseline,
        })

        if args.rl_logging_steps > 0 and (step == 1 or step % args.rl_logging_steps == 0):
            print(json.dumps(row, ensure_ascii=False))

        if args.rl_save_steps > 0 and step % args.rl_save_steps == 0:
            ckpt_path = get_checkpoint_path(args, step, optimized=True)
            save_model(ckpt_path, 'rlmodel', model, optimizer=optimizer, epoch=step, loss_log={'rollout_step': step})

    final_path = get_checkpoint_path(args, args.rl_epochs, optimized=True)
    save_model(final_path, 'rlmodel', model, optimizer=optimizer, epoch=args.rl_epochs, loss_log={'rollout_step': max_steps})
    metrics['final_checkpoint'] = final_path
    return SimpleNamespace(metrics=metrics)


def extract_sample_batch(args, device, sample, tokenizer, src_len, tgt_len, is_gen):
    del args
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
    )


def train_loop(
        args,
        model,
        tokenizer,
        optimizer,
        scheduler,
        dataloader,
        device,
        src_len,
        tgt_len,
        accelerator=None,
        epoch=None,
        total_epochs=None,
        on_log_step=None):
    model.train()
    niter = len(dataloader)
    total_loss = 0.0
    total_steps = 0
    window_losses = []
    effective_total = min(niter, args.max_train_batches) if args.max_train_batches > 0 else niter
    start_time = time.time()

    iterator = enumerate(dataloader, start=1)
    if args.progress_bar:
        iterator = tqdm(
            iterator,
            total=effective_total,
            disable=(accelerator is not None) and (not accelerator.is_local_main_process),
        )

    for step, sample in iterator:
        _, _, _, input_ids, attention_mask, labels, _, _ = extract_sample_batch(
            args=args,
            device=device,
            sample=sample,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gen=False,
        )

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss_value = float(loss.detach().item())
        total_loss += loss_value
        total_steps += 1
        window_losses.append(loss_value)

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        if should_emit_periodic_log(step, effective_total, args.train_log_every):
            elapsed = max(time.time() - start_time, 1e-8)
            avg_loss = total_loss / total_steps
            window_avg = average_or_none(window_losses)
            it_per_sec = step / elapsed
            remaining_steps = max(effective_total - step, 0)
            eta_seconds = remaining_steps / it_per_sec if it_per_sec > 0 else 0
            epoch_label = f'{epoch}/{total_epochs}' if epoch is not None and total_epochs is not None else str(epoch or '?')
            progress_pct = (step / effective_total * 100.0) if effective_total > 0 else 0.0
            print(
                f'[train][epoch {epoch_label}] '
                f'step {step}/{effective_total} ({progress_pct:5.1f}%) '
                f'loss={loss_value:.6f} window_avg={window_avg:.6f} global_avg={avg_loss:.6f} '
                f'lr={scheduler.get_last_lr()[0]:.3e} '
                f'{it_per_sec:.2f} it/s '
                f'elapsed={format_seconds(elapsed)} eta={format_seconds(eta_seconds)}',
                flush=True,
            )
            if on_log_step is not None:
                on_log_step(
                    step=step,
                    effective_total=effective_total,
                    loss_value=loss_value,
                    window_avg=window_avg,
                    global_avg=avg_loss,
                )
            window_losses.clear()

        if args.max_train_batches > 0 and step >= args.max_train_batches:
            break

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_loop(
        args,
        model,
        tokenizer,
        dataloader,
        device,
        src_len,
        tgt_len,
        accelerator=None,
        max_batches=None):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    effective_max_batches = max_batches if max_batches is not None else args.max_valid_batches

    for step, sample in enumerate(dataloader, start=1):
        _, _, _, input_ids, attention_mask, labels, _, _ = extract_sample_batch(
            args=args,
            device=device,
            sample=sample,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gen=False,
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.detach()

        if accelerator is not None:
            gathered_loss = accelerator.gather(loss.reshape(1))
            total_loss += gathered_loss.float().mean().item()
        else:
            total_loss += float(loss.cpu())

        total_steps += 1
        if effective_max_batches > 0 and step >= effective_max_batches:
            break

    return total_loss / max(total_steps, 1)


def get_checkpoint_candidates(args, epoch, optimized=False):
    suffixes = [PIPELINE_TAG]
    if LEGACY_PIPELINE_TAG not in suffixes:
        suffixes.append(LEGACY_PIPELINE_TAG)

    candidates = []
    for suffix in suffixes:
        if optimized:
            filename = f'{args.dataname}-{args.scale}-{args.max_answer_size}-{epoch}-rl-{suffix}.pth'
        else:
            filename = f'{args.dataname}-{args.scale}-{args.max_answer_size}-{epoch}-{suffix}.pth'
        candidates.append(os.path.join(args.checkpoint_root, args.modelname, filename))
    return candidates


def get_checkpoint_path(args, epoch, optimized=False):
    candidates = get_checkpoint_candidates(args, epoch, optimized=optimized)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def get_resume_checkpoint_path(args, epoch, optimized=False):
    explicit_path = str(getattr(args, 'checkpoint_path', '') or '').strip()
    if explicit_path:
        return explicit_path
    return get_checkpoint_path(args, epoch, optimized=optimized)


def load_model_by_mode(args, device, model_name, ntoken, config_train, special_tokens, model_runtime_config=None):
    optimizer = None
    scheduler = None
    last_epoch = 0
    loss_log = {'train': {}, 'valid': {}}

    if args.mode == 'rl' and args.rl_resume_epoch != 0:
        resume_path = get_resume_checkpoint_path(args, args.rl_resume_epoch, optimized=True)
        print(f'Loading RL model: {resume_path}')
        model, optimizer, scheduler, last_epoch, loss_log = load_model(
            resume_path,
            'rlmodel',
            return_huggingface_model=True,
            epoch=args.rl_resume_epoch,
        )
        model.model_name = model_name
        model.to(device)
    elif args.resume_epoch != 0:
        resume_path = get_resume_checkpoint_path(args, args.resume_epoch, optimized=False)
        print(f'Loading model: {resume_path}')
        base_model = None
        if args.use_peft:
            print('# Creating base model before loading PEFT adapter checkpoint.')
            base_model = create_transformer(
                ntoken=ntoken,
                special_tokens=special_tokens,
                model_name=model_name,
                vocab_size=ntoken,
                use_pretrained_weights=args.use_pretrained_text_model,
                model_runtime_config=model_runtime_config,
            )
        model, optimizer, scheduler, last_epoch, loss_log = load_model(
            resume_path,
            'model',
            return_huggingface_model=True,
            epoch=args.resume_epoch,
            model=base_model,
        )
        model.model_name = model_name
        model.to(device)
    else:
        print('Creating model')
        model = create_transformer(
            ntoken=ntoken,
            special_tokens=special_tokens,
            model_name=model_name,
            vocab_size=ntoken,
            use_pretrained_weights=args.use_pretrained_text_model,
            model_runtime_config=model_runtime_config,
        ).to(device)

    newly_wrapped_lora = False
    if args.mode == 'training':
        model, newly_wrapped_lora = apply_lora_if_requested(model, args)
        model.to(device)
        if optimizer is None or scheduler is None or newly_wrapped_lora:
            optimizer, scheduler = create_optimizer_and_scheduler(model, config_train)

    if args.mode == 'rl' and args.rl_resume_epoch == 0 and args.rl_use_peft:
        original_use_peft = args.use_peft
        args.use_peft = True
        model, _ = apply_lora_if_requested(model, args)
        args.use_peft = original_use_peft
        model.to(device)

    print('model.config:')
    print(model.config)

    if args.mode == 'training':
        return model, optimizer, scheduler, last_epoch, loss_log
    return model


def fit(
        args,
        nepoch,
        dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
        model_name,
        src_len,
        tgt_len,
        last_epoch,
        loss_log,
        device,
        accelerator,
        dataset_dict,
        kg,
        experiment_record=None):
    train_dataloader = dataloader['train']
    valid_dataloader = dataloader.get('valid')

    if accelerator is not None:
        if valid_dataloader is None:
            model, optimizer, train_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, scheduler
            )
        else:
            model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader, scheduler
            )

    result_path = os.path.join(
        args.result_root,
        args.modelname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}_results.txt',
    )
    pathlib.Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(last_epoch + 1, nepoch + 1):
        print('lr:', scheduler.get_last_lr())

        def on_train_log_step(step, effective_total, loss_value, window_avg, global_avg):
            if valid_dataloader is not None and args.intra_epoch_eval_every > 0 and step % args.intra_epoch_eval_every == 0:
                snapshot_valid = evaluate_loop(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=valid_dataloader,
                    device=device,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    accelerator=accelerator,
                    max_batches=args.intra_epoch_eval_batches,
                )
                msg = (
                    f'[valid-snapshot][epoch {epoch}/{nepoch}] '
                    f'step {step}/{effective_total}'
                    f'train_window_avg={window_avg:.6f} '
                    f'valid_snapshot={snapshot_valid:.6f} '
                    f'valid_batches={args.intra_epoch_eval_batches if args.intra_epoch_eval_batches > 0 else "all"}'
                )
                logging.info(msg)
                print(msg, flush=True)
                if experiment_record is not None:
                    append_text_log(experiment_record['paths']['run_log_path'], msg)

            if (
                    experiment_record is not None
                    and args.intra_epoch_comparison_every > 0
                    and step % args.intra_epoch_comparison_every == 0):
                log_prediction_comparisons(
                    args=args,
                    dataset_dict=dataset_dict,
                    model=model,
                    tokenizer=tokenizer,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    device=device,
                    accelerator=accelerator,
                    log_path=experiment_record['paths']['comparison_log_path'],
                    stage_label=f'epoch_{epoch}_step_{step}',
                )

        loss_train = train_loop(
            args=args,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
            device=device,
            src_len=src_len,
            tgt_len=tgt_len,
            accelerator=accelerator,
            epoch=epoch,
            total_epochs=nepoch,
            on_log_step=on_train_log_step,
        )
        loss_log['train'][epoch] = loss_train

        loss_valid = None
        if valid_dataloader is not None:
            loss_valid = evaluate_loop(
                args=args,
                model=model,
                tokenizer=tokenizer,
                dataloader=valid_dataloader,
                device=device,
                src_len=src_len,
                tgt_len=tgt_len,
                accelerator=accelerator,
            )
            loss_log['valid'][epoch] = loss_valid

        msg = f'epoch: {epoch}, train loss: {loss_train}'
        if loss_valid is not None:
            msg += f', valid loss: {loss_valid}'
        if epoch > 1 and (epoch - 1) in loss_log['train']:
            train_delta = loss_train - loss_log['train'][epoch - 1]
            msg += f', train delta: {train_delta:+.6f}'
        if loss_valid is not None and (epoch - 1) in loss_log['valid']:
            valid_delta = loss_valid - loss_log['valid'][epoch - 1]
            msg += f', valid delta: {valid_delta:+.6f}'
        logging.info(msg)
        print(f'[epoch-summary] {msg}', flush=True)

        with open(result_path, 'a', encoding='utf-8') as result_file:
            result_file.write(msg + '\n')

        if experiment_record is not None:
            append_text_log(experiment_record['paths']['run_log_path'], msg)
            append_csv_row(
                experiment_record['paths']['loss_csv_path'],
                ['epoch', 'train_loss', 'valid_loss'],
                {
                    'epoch': epoch,
                    'train_loss': f'{loss_train:.6f}',
                    'valid_loss': '' if loss_valid is None else f'{loss_valid:.6f}',
                },
            )
            if args.comparison_frequency > 0 and epoch % args.comparison_frequency == 0:
                log_prediction_comparisons(
                    args=args,
                    dataset_dict=dataset_dict,
                    model=model,
                    tokenizer=tokenizer,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    device=device,
                    accelerator=accelerator,
                    log_path=experiment_record['paths']['comparison_log_path'],
                    stage_label=f'epoch_{epoch}',
                )

        if epoch % args.save_frequency == 0 or epoch == nepoch:
            ckpt_path = get_checkpoint_path(args, epoch, optimized=False)
            model_to_save = accelerator.unwrap_model(model) if accelerator is not None else model
            optimizer_to_save = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer
            scheduler_to_save = scheduler
            save_model(ckpt_path, 'model', model_to_save, optimizer_to_save, scheduler_to_save, epoch, loss_log)

        print('=' * 50)

    if experiment_record is not None:
        log_prediction_comparisons(
            args=args,
            dataset_dict=dataset_dict,
            model=model,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            device=device,
            accelerator=accelerator,
            log_path=experiment_record['paths']['comparison_log_path'],
            stage_label='final',
        )
        write_experiment_summary(experiment_record=experiment_record, loss_log=loss_log)


def mask_source(device, source_attention_mask, pred, tokenizer):
    batch_size = pred.shape[0]
    diff = pred.shape[-1] - source_attention_mask.shape[-1]
    prefix_mask = torch.cat(
        [
            source_attention_mask,
            torch.zeros((batch_size, diff), dtype=torch.bool, device=device),
        ],
        dim=1,
    ).to(device)
    pred[prefix_mask == 1] = tokenizer.pad_token_id


def save_model(path, contents, model, optimizer=None, scheduler=None, epoch=None, loss_log=None):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if contents in {'model', 'rlmodel'}:
        print(f'# Saving checkpoint ({contents}) {path}')
        if is_peft_model(model):
            adapter_dir = f'{path}.adapter'
            if os.path.exists(adapter_dir):
                shutil.rmtree(adapter_dir)
            model.save_pretrained(adapter_dir)
            checkpoint = {
                'peft_checkpoint': True,
                'peft_adapter_dir': os.path.basename(adapter_dir),
                'epoch': epoch,
                'loss_log': loss_log,
            }
            if optimizer is not None and hasattr(optimizer, 'state_dict'):
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None and hasattr(scheduler, 'state_dict'):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, path)
            return
        torch.save(
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'epoch': epoch,
                'loss_log': loss_log,
            },
            path,
        )
        return
    raise ValueError(f'Unsupported contents: {contents}')


def my_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelname', default='GPT2_6_act_nt')
    parser.add_argument('--config-model', default='configs/config-model.yml')
    parser.add_argument('--config-dataloader', default='configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='configs/config-train.yml')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_root', default='./sampled_data_abduction/')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('-a', '--max-answer-size', type=int, default=8)
    parser.add_argument('--scale', default='default')

    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument(
        '--checkpoint-path',
        dest='checkpoint_path',
        default='',
        help='Explicit checkpoint path for loading/resuming. Saving still uses --checkpoint_root.',
    )
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)
    parser.add_argument('--use_pretrained_text_model', action='store_true')
    parser.add_argument(
        '--disable_text_extra_tokens',
        action='store_true',
        help='Do not add ACTION/DSL/PATTERN as new tokenizer tokens; useful for lightweight LoRA on pretrained LMs.',
    )
    parser.add_argument('--use_peft', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', default='none', choices=['none', 'all', 'lora_only'])
    parser.add_argument(
        '--lora_target_modules',
        default='',
        help='Comma-separated LoRA target modules. Empty means infer from model type.',
    )
    parser.add_argument(
        '--lora_modules_to_save',
        default='auto',
        help='Comma-separated extra trainable modules to save with LoRA. Use auto/none.',
    )

    parser.add_argument('--mode', default='training', choices=['training', 'rl'])
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--mixed_precision', default='no', choices=['no', 'fp16', 'bf16'])

    parser.add_argument('--result_root', default='./results/')
    parser.add_argument('--save_frequency', type=int, default=1)

    parser.add_argument('--max_train_rows', type=int, default=0)
    parser.add_argument('--max_valid_rows', type=int, default=0)
    parser.add_argument('--max_train_batches', type=int, default=0)
    parser.add_argument('--max_valid_batches', type=int, default=0)
    parser.add_argument('--override_nepoch', type=int, default=0)
    parser.add_argument('--override_lr', type=float, default=0.0)
    parser.add_argument('--override_warm_up', type=int, default=0)

    parser.add_argument('--experiment_root', default='./results/experiments/')
    parser.add_argument('--rl_experiment_root', default='./results/rl_experiments/')
    parser.add_argument('--experiment_name', default='')
    parser.add_argument('--comparison_samples', type=int, default=3)
    parser.add_argument('--comparison_frequency', type=int, default=1)
    parser.add_argument('--comparison_console', type=str2bool, default=True)
    parser.add_argument('--comparison_random', type=str2bool, default=False)
    parser.add_argument('--train_log_every', type=int, default=5000)
    parser.add_argument('--progress_bar', type=str2bool, default=False)
    parser.add_argument('--intra_epoch_eval_every', type=int, default=5000)
    parser.add_argument('--intra_epoch_eval_batches', type=int, default=16)
    parser.add_argument('--intra_epoch_comparison_every', type=int, default=20000)

    parser.add_argument('--dataset_cache_root', default='./dataset_cache/')
    parser.add_argument('--dataset_num_proc', type=int, default=1)
    parser.add_argument('--dataset_map_batch_size', type=int, default=1000)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--dataloader_pin_memory', type=str2bool, default=False)
    parser.add_argument('--dataloader_persistent_workers', type=str2bool, default=True)
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=2)
    parser.add_argument('--force_load_kg', action='store_true')
    parser.add_argument(
        '--train_stage',
        default='logic',
        choices=['logic', 'stage2'],
    )
    parser.add_argument('--result_top_k', type=int, default=3)

    parser.add_argument('--pattern_path', type=str, default='./metadata/pattern_filtered.csv')

    parser.add_argument('--rl_resume_epoch', type=int, default=0)
    parser.add_argument('--rl_proportion', type=float, default=1.0)
    parser.add_argument('--rl_epochs', type=int, default=4)
    parser.add_argument('--rl_search_split', default='train')
    parser.add_argument('--rl_lr', type=float, default=1e-6)
    parser.add_argument('--rl_use_peft', action='store_true')
    parser.add_argument('--rl_max_steps', type=int, default=-1)
    parser.add_argument('--rl_max_prompt_length', type=int, default=128)
    parser.add_argument('--rl_max_completion_length', type=int, default=128)
    parser.add_argument('--rl_max_action_steps', type=int, default=3)
    parser.add_argument('--rl_temperature', type=float, default=0.8)
    parser.add_argument('--rl_top_k', type=int, default=50)
    parser.add_argument('--rl_top_p', type=float, default=1.0)
    parser.add_argument('--rl_baseline_momentum', type=float, default=0.9)
    parser.add_argument('--rl_advantage_clip', type=float, default=2.0)
    parser.add_argument('--rl_grad_clip', type=float, default=1.0)
    parser.add_argument('--rl_logging_steps', type=int, default=10)
    parser.add_argument('--rl_save_steps', type=int, default=100)

    parser.add_argument('--MAX_STAGE1_BATCHES', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--MAX_STAGE2_BATCHES', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--MAX_VALID_BATCHES', type=int, default=None, help=argparse.SUPPRESS)

    args, unknown_args = parser.parse_known_args()
    remaining_unknown = [arg for arg in unknown_args if arg != '\\']
    if remaining_unknown:
        parser.error(f'unrecognized arguments: {" ".join(remaining_unknown)}')

    if args.train_stage == 'logic' and args.MAX_STAGE1_BATCHES is not None:
        args.max_train_batches = args.MAX_STAGE1_BATCHES
    if args.train_stage == 'stage2' and args.MAX_STAGE2_BATCHES is not None:
        args.max_train_batches = args.MAX_STAGE2_BATCHES
    if args.MAX_VALID_BATCHES is not None:
        args.max_valid_batches = args.MAX_VALID_BATCHES

    return args


def main():
    args = my_parse_args()
    print(f'args:\n{args}\n')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.join(args.result_root, args.modelname), exist_ok=True)

    config_dataloader = load_yaml(args.config_dataloader)
    print(f'config_dataloader:\n{config_dataloader}\n')
    config_model = load_yaml(args.config_model)
    model_runtime_config = resolve_model_runtime_config(args.modelname, config_model)
    print(f'model_runtime_config:\n{model_runtime_config}\n')

    pattern_filtered = pd.read_csv(args.pattern_path, index_col='id')

    if args.accelerate and args.mode != 'rl':
        if Accelerator is None:
            raise ImportError('accelerate is not installed. Please install it or run without --accelerate.')
        mixed_precision = None if args.mixed_precision == 'no' else args.mixed_precision
        accelerator = Accelerator(mixed_precision=mixed_precision)
        device = accelerator.device
    else:
        accelerator = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    model_name = args.modelname
    src_len = config_dataloader['text_obs_len']
    tgt_len = config_dataloader['text_hyp_len']
    print(f'model_name:{model_name}\n')
    print(f'batch_size:{args.batch_size}\n')
    print('=' * 50)

    if args.mode == 'training':
        splits = ['train']
        try:
            resolve_sampled_dataset_path(args.data_root, args.dataname, 'valid')
            splits.append('valid')
        except FileNotFoundError:
            print('# Warning: valid split not found yet, training will use train split only.')
    else:
        splits = ['train']
        if args.rl_search_split not in splits:
            splits.append(args.rl_search_split)

    if training_data_needs_kg(args, splits):
        print('Loading graph')
        kg = load_kg(args.dataname)
        graph_samplers = kg.graph_samplers
    else:
        print('# Skipping KG load: training data already has logic_dsl/stage2_trace.')
        kg = None
        graph_samplers = None

    dataset_train_stage = args.train_stage
    if args.mode == 'rl' and args.train_stage == 'stage2':
        # Rollout RL starts from raw OBS and lets the model decide ACTION/DSL.
        # Do not expand Stage 2 SFT prefixes for the RL dataset.
        dataset_train_stage = 'logic'

    print('Creating dataset & dataloader')
    dataset_dict, _, _ = new_create_dataset(
        dataname=args.dataname,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        max_rows_by_split={
            'train': args.max_train_rows,
            'valid': args.max_valid_rows,
        },
        kg=kg,
        source_text_field=DEFAULT_SOURCE_TEXT_FIELD,
        target_text_field=DEFAULT_TARGET_TEXT_FIELD,
        representation='text',
        dataset_cache_root=args.dataset_cache_root,
        dataset_num_proc=args.dataset_num_proc,
        dataset_map_batch_size=args.dataset_map_batch_size,
        train_stage=dataset_train_stage,
        result_top_k=args.result_top_k,
    )

    if args.mode == 'rl' and args.rl_proportion < 1:
        nrows = dataset_dict['train'].shape[0]
        selected = random.sample(range(nrows), int(nrows * args.rl_proportion))
        dataset_dict['train'] = dataset_dict['train'].select(selected)

    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=args.batch_size,
        drop_last=(args.mode == 'rl'),
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )

    print('Creating tokenizer')
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

    all_train_configs = load_yaml(args.config_train)
    config_train = all_train_configs.get(model_name, all_train_configs['default'])
    if args.override_nepoch > 0:
        config_train = dict(config_train)
        config_train['nepoch'] = args.override_nepoch
    if args.override_lr > 0:
        config_train = dict(config_train)
        config_train['lr'] = args.override_lr
    if args.override_warm_up > 0:
        config_train = dict(config_train)
        config_train['warm_up'] = args.override_warm_up
    print(f'config_train:\n{config_train}')

    experiment_record = None
    if args.mode == 'training':
        experiment_record = prepare_experiment_record(
            args=args,
            dataset_dict=dataset_dict,
            config_train=config_train,
            config_dataloader=config_dataloader,
            device=device,
        )
    elif args.mode == 'rl':
        experiment_record = prepare_rl_experiment_record(
            args=args,
            dataset_dict=dataset_dict,
            device=device,
        )

    if args.mode == 'training':
        model, optimizer, scheduler, last_epoch, loss_log = load_model_by_mode(
            args=args,
            device=device,
            model_name=model_name,
            ntoken=ntoken,
            config_train=config_train,
            special_tokens=special_tokens,
            model_runtime_config=model_runtime_config,
        )
    else:
        model = load_model_by_mode(
            args=args,
            device=device,
            model_name=model_name,
            ntoken=ntoken,
            config_train=config_train,
            special_tokens=special_tokens,
            model_runtime_config=model_runtime_config,
        )

    if args.mode == 'training':
        fit(
            args=args,
            nepoch=config_train['nepoch'],
            dataloader=dataloader_dict,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            model_name=model_name,
            src_len=src_len,
            tgt_len=tgt_len,
            last_epoch=last_epoch,
            loss_log=loss_log,
            device=device,
            accelerator=accelerator if args.accelerate else None,
            dataset_dict=dataset_dict,
            kg=kg,
            experiment_record=experiment_record,
        )
    else:
        log_prediction_comparisons(
            args=args,
            dataset_dict=dataset_dict,
            model=model,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            device=device,
            accelerator=None,
            log_path=experiment_record['paths']['comparison_log_path'],
            stage_label='before_rl',
        )
        trainer_result = optimize_rollout_policy(
            args=args,
            dataset=dataset_dict['train'],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            kg=kg,
            experiment_record=experiment_record,
        )
        log_prediction_comparisons(
            args=args,
            dataset_dict=dataset_dict,
            model=model,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            device=device,
            accelerator=None,
            log_path=experiment_record['paths']['comparison_log_path'],
            stage_label='after_rl',
        )
        write_rl_experiment_summary(experiment_record, trainer_result)


if __name__ == '__main__':
    main()
