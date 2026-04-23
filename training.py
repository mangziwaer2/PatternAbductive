import argparse
import csv
import datetime
import inspect
import json
import logging
import os
import pathlib
import platform
import random
import subprocess
import sys
import time

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
    source_to_prompt,
)
from model.transformer import create_transformer, GPT2_MODEL_PATH
from utils.condition import DEFAULT_EXCLUDED_CONDITION_TYPES, normalize_condition_type_list
from utils.dataloader import (
    filter_dataset_by_excluded_condition_types,
    new_create_dataloader,
    new_create_dataset,
)
from utils.kg_hints import build_batch_kg_hints_texts
from utils.load import load_kg, load_model, load_yaml, resolve_sampled_dataset_path
from utils.stat_util import stat_scores_by_pattern
from utils.text_scoring import score_text_query_batch


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


def collect_dataset_sizes(dataset_dict):
    return {
        split: int(dataset.shape[0])
        for split, dataset in dataset_dict.items()
    }


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
            'pipeline': 'OBS + COND + KG_HINTS -> hypothesis_text',
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
        f'# comparison_samples={args.comparison_samples}, comparison_frequency={args.comparison_frequency}',
    )
    return {
        'name': experiment_name,
        'paths': paths,
        'metadata': metadata,
    }


def prepare_rl_experiment_record(args, dataset_dict, device):
    experiment_name = build_experiment_name(args)
    experiment_dir = os.path.join(args.optim_experiment_root, experiment_name)
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
            'pipeline': 'OBS + COND + KG_HINTS -> hypothesis_text',
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
        f'# comparison_samples={args.comparison_samples}, comparison_frequency={args.comparison_frequency}',
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
        '## GRPO Metrics',
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


def select_sample_indices(dataset_size, num_samples):
    if dataset_size <= 0 or num_samples <= 0:
        return []
    if num_samples >= dataset_size:
        return list(range(dataset_size))
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


def build_logged_input(source_value, condition_value, kg_hints_value=''):
    condition_value = format_logged_condition(condition_value).strip()
    kg_hints_value = format_logged_condition(kg_hints_value).strip()
    parts = [source_value]
    if condition_value:
        parts.extend(['SEP', condition_value])
    if kg_hints_value:
        parts.extend(['SEP', kg_hints_value])
    return ' '.join(parts)


def should_use_kg_hints(args):
    return args.use_kg_hints and args.kg_hints_max_facts > 0


def maybe_build_batch_kg_hints(args, sample, kg, kg_hint_split):
    if 'kg_hints_text' in sample:
        cached_hints = sample.get('kg_hints_text')
        if cached_hints is not None:
            cached_hints = [str(value).strip() for value in cached_hints]
            if any(cached_hints):
                return cached_hints

    if not should_use_kg_hints(args) or kg is None:
        return None

    source = sample.get('source')
    if source is None:
        return None

    condition_texts = sample.get(DEFAULT_CONDITION_TEXT_FIELD, [''] * len(source))
    return build_batch_kg_hints_texts(
        observation_texts=source,
        kg=kg,
        condition_texts=condition_texts,
        graph_split=kg_hint_split,
        max_facts=args.kg_hints_max_facts,
    )


def run_generation(model, input_ids, attention_mask, tokenizer, max_length, top_k=0, do_sample=True):
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
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
        stage_label,
        kg=None):
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
        indices = select_sample_indices(len(dataset), args.comparison_samples)
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
                    kg=kg,
                    kg_hint_split=split,
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

            batch_hints = maybe_build_batch_kg_hints(args, sample, kg, split)
            kg_hints_value = batch_hints[0] if batch_hints else ''
            condition_value = condition[0] if condition else ''

            emit_text_log(f'[{split}] idx={sample_index} pattern_id={pattern_id[0]}', log_path, also_print=args.comparison_console)
            emit_text_log(
                f'[{split}] INPUT  : {build_logged_input(source[0], condition_value, kg_hints_value)}',
                log_path,
                also_print=args.comparison_console,
            )
            emit_text_log(f'[{split}] TARGET : {target[0]}', log_path, also_print=args.comparison_console)
            emit_text_log(f'[{split}] PRED   : {prediction}', log_path, also_print=args.comparison_console)
            emit_text_log('', log_path, also_print=args.comparison_console)


def parse_rl_factors(raw_value):
    if isinstance(raw_value, (list, tuple)):
        factors = [float(value) for value in raw_value]
    else:
        text = str(raw_value).strip()
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError
            factors = [float(value) for value in parsed]
        except Exception:
            factors = [float(token.strip()) for token in text.split(',') if token.strip()]

    while len(factors) < 4:
        factors.append(1.0)
    return factors[:4]


def build_grpo_dataset(dataset, args, kg, kg_hint_split='train'):
    dataset = dataset.map(
        lambda example: source_to_prompt(
            example,
            args=args,
            kg=kg,
            kg_hint_split=kg_hint_split,
        )
    )
    keep_columns = ['prompt', 'source', 'target', DEFAULT_CONDITION_TEXT_FIELD]
    removable = [column for column in dataset.column_names if column not in keep_columns]
    if removable:
        dataset = dataset.remove_columns(removable)
    return dataset


def optimize_grpo(args, dataset, model, tokenizer, graph_samplers, kg, batch_size, experiment_record=None):
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError('TRL is required for optimizing mode. Please install `trl`.') from exc

    dataset = build_grpo_dataset(dataset, args, kg=kg, kg_hint_split='train')
    output_dir = (
        experiment_record['paths']['experiment_dir']
        if experiment_record is not None
        else f'./results/optim/{build_experiment_name(args)}'
    )
    report_to = None if str(args.rl_report_to).lower() in {'', 'none', 'null'} else args.rl_report_to
    rl_factors = parse_rl_factors(args.rl_factor)

    def reward_func(prompts, completions, target, source, condition_text=None, **kwargs):
        condition_texts = condition_text or [''] * len(completions)
        scores = score_text_query_batch(
            completions=completions,
            targets=target,
            sources=source,
            condition_texts=condition_texts,
            kg=kg,
            graph_samplers=graph_samplers,
            searching_split=args.rl_search_split,
        )
        return [
            float(
                score['jaccard'] * rl_factors[0]
                + score['dice'] * rl_factors[1]
                + score['overlap'] * rl_factors[2]
                + score['condition'] * rl_factors[3]
            )
            for score in scores
        ]

    grpo_config = GRPOConfig(
        seed=args.seed,
        output_dir=output_dir,
        num_train_epochs=args.rl_epochs,
        max_steps=args.rl_max_steps,
        learning_rate=args.rl_lr,
        max_prompt_length=args.rl_max_prompt_length,
        max_completion_length=args.rl_max_completion_length,
        num_generations=args.rl_num_generations,
        logging_steps=args.rl_logging_steps,
        save_steps=args.rl_save_steps,
        log_completions=args.rl_log_completions,
        report_to=report_to,
        beta=args.rl_init_kl_coef,
        epsilon=args.rl_cliprange,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
    )
    print(grpo_config)

    original_forward = None
    if 'logits_to_keep' not in inspect.signature(model.forward).parameters:
        original_forward = model.forward

        def forward_with_grpo_compat(*forward_args, **forward_kwargs):
            forward_kwargs.pop('logits_to_keep', None)
            return original_forward(*forward_args, **forward_kwargs)

        model.forward = forward_with_grpo_compat

    model.warnings_issued = {}
    original_add_model_tags = getattr(model, 'add_model_tags', None)

    def dummy_add_model_tags(self, tags):
        return None

    model.add_model_tags = dummy_add_model_tags.__get__(model)
    trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer_result = trainer.train()
    if original_forward is not None:
        model.forward = original_forward
    if original_add_model_tags is not None:
        model.add_model_tags = original_add_model_tags
    elif 'add_model_tags' in model.__dict__:
        del model.__dict__['add_model_tags']

    trainer.save_model(output_dir)
    ckpt_path = get_checkpoint_path(args, args.rl_epochs, optimized=True)
    save_model(ckpt_path, 'model', model, epoch=args.rl_epochs)
    if experiment_record is not None:
        append_text_log(
            experiment_record['paths']['run_log_path'],
            json.dumps(trainer_result.metrics, ensure_ascii=False, indent=2),
        )
    return trainer_result


def extract_sample_batch(args, device, sample, tokenizer, src_len, tgt_len, is_gen, kg=None, kg_hint_split='train'):
    kg_hints_text = maybe_build_batch_kg_hints(args, sample, kg, kg_hint_split)
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
        kg_hints_text=kg_hints_text,
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
        on_log_step=None,
        kg=None,
        kg_hint_split='train'):
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
            kg=kg,
            kg_hint_split=kg_hint_split,
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
        max_batches=None,
        kg=None,
        kg_hint_split='valid'):
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
            kg=kg,
            kg_hint_split=kg_hint_split,
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
            filename = f'{args.dataname}-{args.scale}-{args.max_answer_size}-{epoch}-optimize-{suffix}.pth'
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


def load_model_by_mode(args, device, model_name, ntoken, config_train, special_tokens):
    optimizer = None
    scheduler = None
    last_epoch = 0
    loss_log = {'train': {}, 'valid': {}}

    if args.mode in ['optimizing', 'testing'] and args.rl_resume_epoch != 0:
        resume_path = get_checkpoint_path(args, args.rl_resume_epoch, optimized=True)
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
        resume_path = get_checkpoint_path(args, args.resume_epoch, optimized=False)
        print(f'Loading model: {resume_path}')
        model, optimizer, scheduler, last_epoch, loss_log = load_model(
            resume_path,
            'model',
            return_huggingface_model=True,
            epoch=args.resume_epoch,
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
        ).to(device)
        if args.mode == 'training':
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config_train['lr']))
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=config_train['warm_up'],
            )

    if args.mode == 'optimizing' and args.rl_resume_epoch == 0 and args.rl_use_peft:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, peft_config=lora_config)

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
                    kg=kg,
                    kg_hint_split='valid',
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
                    kg=kg,
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
            kg=kg,
            kg_hint_split='train',
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
                kg=kg,
                kg_hint_split='valid',
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
                    kg=kg,
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
            kg=kg,
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


@torch.no_grad()
def test_loop(
        args,
        dataloader,
        model,
        tokenizer,
        graph_samplers,
        pattern_filtered,
        searching_split,
        resume_epoch,
        src_len,
        tgt_len,
        kg,
        device,
        accelerator):
    score_file_suffix = f'test|{args.test_proportion}x{args.test_split}_topk{args.test_top_k}'
    if args.rl_resume_epoch != 0:
        score_file_suffix += f'|grpo-{args.rl_resume_epoch}'
    score_file_suffix = sanitize_filename_component(score_file_suffix)

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    niter = len(dataloader)
    scores_all = []
    pattern_id_all = []
    score_df = None
    do_sample = args.test_top_k > 0

    import torch.distributed as dist

    with torch.no_grad():
        for _, sample in (pbar := tqdm(
                enumerate(dataloader, start=1),
                total=niter,
                disable=(accelerator is not None) and (not accelerator.is_local_main_process))):
            source, target, pattern_id, input_ids, attention_mask, _, source_attention_mask, condition = \
                extract_sample_batch(
                    args=args,
                    device=device,
                    sample=sample,
                    tokenizer=tokenizer,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    is_gen=True,
                    kg=kg,
                    kg_hint_split=searching_split,
                )

            pred = run_generation(
                model=accelerator.unwrap_model(model) if accelerator is not None else model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                max_length=input_ids.shape[1] + tgt_len,
                top_k=args.test_top_k,
                do_sample=do_sample,
            )
            mask_source(device, source_attention_mask, pred, tokenizer)
            pred_decoded = [decode_text_token_ids(tokenizer, sequence.tolist()) for sequence in pred]

            scores = score_text_query_batch(
                completions=pred_decoded,
                targets=target,
                sources=source,
                condition_texts=condition,
                kg=kg,
                graph_samplers=graph_samplers,
                searching_split=searching_split,
            )

            if accelerator is not None:
                gathered_scores = [None] * accelerator.num_processes
                gathered_pattern_id = [None] * accelerator.num_processes
                dist.all_gather_object(gathered_scores, scores)
                dist.all_gather_object(gathered_pattern_id, list(pattern_id))
                gathered_scores = [item for chunk in gathered_scores for item in chunk]
                gathered_pattern_id = [item for chunk in gathered_pattern_id for item in chunk]
            else:
                gathered_scores = scores
                gathered_pattern_id = list(pattern_id)

            if accelerator is None or accelerator.is_main_process:
                scores_all.extend(gathered_scores)
                pattern_id_all.extend(gathered_pattern_id)
                score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                pbar.set_description(
                    f's: {round(score_df.loc["all", ("smatch", "mean")], 4)}, '
                    f'j: {round(score_df.loc["all", ("jaccard", "mean")], 4)}'
                )
                scores_path = os.path.join(
                    args.result_root,
                    args.modelname,
                    f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-scores({score_file_suffix}).csv',
                )
                score_df.to_csv(scores_path)

    if score_df is None:
        raise RuntimeError('Test dataloader is empty; no scores were produced.')
    return score_df


def save_model(path, contents, model, optimizer=None, scheduler=None, epoch=None, loss_log=None):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if contents == 'model':
        print(f'# Saving checkpoint (model) {path}')
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
    parser.add_argument('--config-dataloader', default='configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='configs/config-train.yml')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_root', default='./sampled_data_surface/')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('--scale', default='default')

    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)
    parser.add_argument('--use_pretrained_text_model', action='store_true')

    parser.add_argument('--mode', default='training', choices=['training', 'testing', 'optimizing'])
    parser.add_argument('--accelerate', action='store_true')

    parser.add_argument('--test_proportion', type=float, default=1.0)
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--test_top_k', type=int, default=0)

    parser.add_argument('--result_root', default='./results/')
    parser.add_argument('--save_frequency', type=int, default=1)

    parser.add_argument('--max_train_rows', type=int, default=0)
    parser.add_argument('--max_valid_rows', type=int, default=0)
    parser.add_argument('--max_train_batches', type=int, default=0)
    parser.add_argument('--max_valid_batches', type=int, default=0)
    parser.add_argument('--override_nepoch', type=int, default=0)

    parser.add_argument('--experiment_root', default='./results/experiments/')
    parser.add_argument('--optim_experiment_root', default='./results/optim_experiments/')
    parser.add_argument('--experiment_name', default='')
    parser.add_argument('--comparison_samples', type=int, default=3)
    parser.add_argument('--comparison_frequency', type=int, default=1)
    parser.add_argument('--comparison_console', type=str2bool, default=True)
    parser.add_argument('--train_log_every', type=int, default=5000)
    parser.add_argument('--progress_bar', type=str2bool, default=False)
    parser.add_argument('--intra_epoch_eval_every', type=int, default=5000)
    parser.add_argument('--intra_epoch_eval_batches', type=int, default=16)
    parser.add_argument('--intra_epoch_comparison_every', type=int, default=20000)

    parser.add_argument('--dataset_cache_root', default='./dataset_cache/')
    parser.add_argument('--dataset_num_proc', type=int, default=1)
    parser.add_argument('--dataset_map_batch_size', type=int, default=1000)

    parser.add_argument('--pattern_path', type=str, default='./metadata/pattern_filtered.csv')
    parser.add_argument('--use_kg_hints', type=str2bool, default=True)
    parser.add_argument('--kg_hints_max_facts', type=int, default=8)
    parser.add_argument(
        '--exclude_condition_types',
        default=','.join(sorted(DEFAULT_EXCLUDED_CONDITION_TYPES)),
    )

    parser.add_argument('--rl_resume_epoch', type=int, default=0)
    parser.add_argument('--rl_proportion', type=float, default=1.0)
    parser.add_argument('--rl_epochs', type=int, default=4)
    parser.add_argument('--rl_search_split', default='train')
    parser.add_argument('--rl_lr', type=float, default=1e-6)
    parser.add_argument('--rl_use_peft', action='store_true')
    parser.add_argument('--rl_factor', type=str, default='[1.0, 1.0, 1.0, 1.0]')
    parser.add_argument('--rl_max_steps', type=int, default=-1)
    parser.add_argument('--rl_num_generations', type=int, default=4)
    parser.add_argument('--rl_max_prompt_length', type=int, default=128)
    parser.add_argument('--rl_max_completion_length', type=int, default=128)
    parser.add_argument('--rl_logging_steps', type=int, default=10)
    parser.add_argument('--rl_save_steps', type=int, default=100)
    parser.add_argument('--rl_log_completions', action='store_true')
    parser.add_argument('--rl_report_to', default='none')
    parser.add_argument('--rl_init_kl_coef', type=float, default=0.2)
    parser.add_argument('--rl_cliprange', type=float, default=0.2)

    return parser.parse_args()


def main():
    args = my_parse_args()
    args.exclude_condition_types = normalize_condition_type_list(args.exclude_condition_types)
    print(f'args:\n{args}\n')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.join(args.result_root, args.modelname), exist_ok=True)

    config_dataloader = load_yaml(args.config_dataloader)
    print(f'config_dataloader:\n{config_dataloader}\n')

    pattern_filtered = pd.read_csv(args.pattern_path, index_col='id')

    print('Loading graph')
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers

    if args.accelerate and args.mode != 'optimizing':
        if Accelerator is None:
            raise ImportError('accelerate is not installed. Please install it or run without --accelerate.')
        accelerator = Accelerator()
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
    elif args.mode == 'testing':
        splits = [args.test_split]
    else:
        splits = ['train']
        try:
            resolve_sampled_dataset_path(args.data_root, args.dataname, 'valid')
            splits.append('valid')
        except FileNotFoundError:
            pass
        if args.rl_search_split not in splits:
            splits.append(args.rl_search_split)

    print('Creating dataset & dataloader')
    dataset_dict, _, _ = new_create_dataset(
        dataname=args.dataname,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        max_rows_by_split={
            'train': args.max_train_rows,
            'valid': args.max_valid_rows,
            args.test_split: args.max_valid_rows if args.mode == 'testing' else 0,
        },
        kg=kg,
        source_text_field=DEFAULT_SOURCE_TEXT_FIELD,
        target_text_field=DEFAULT_TARGET_TEXT_FIELD,
        representation='text',
        dataset_cache_root=args.dataset_cache_root,
        dataset_num_proc=args.dataset_num_proc,
        dataset_map_batch_size=args.dataset_map_batch_size,
    )

    # if args.exclude_condition_types:
    #     for split in list(dataset_dict.keys()):
    #         before_count = len(dataset_dict[split])
    #         dataset_dict[split] = filter_dataset_by_excluded_condition_types(
    #             dataset_dict[split],
    #             excluded_condition_types=args.exclude_condition_types,
    #         )
    #         after_count = len(dataset_dict[split])
    #         if after_count != before_count:
    #             print(
    #                 f'# Filtered split "{split}" by excluded condition types '
    #                 f'{args.exclude_condition_types}: {before_count} -> {after_count}'
    #             )

    if args.mode == 'testing' and args.test_proportion < 1:
        nrows = dataset_dict[args.test_split].shape[0]
        selected = random.sample(range(nrows), int(nrows * args.test_proportion))
        dataset_dict[args.test_split] = dataset_dict[args.test_split].select(selected)

    if args.mode == 'optimizing' and args.rl_proportion < 1:
        nrows = dataset_dict['train'].shape[0]
        selected = random.sample(range(nrows), int(nrows * args.rl_proportion))
        dataset_dict['train'] = dataset_dict['train'].select(selected)

    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=args.batch_size,
        drop_last=(args.mode == 'optimizing'),
    )

    print('Creating tokenizer')
    tokenizer, ntoken = create_text_tokenizer(
        GPT2_MODEL_PATH,
        extra_tokens=get_text_extra_tokens(include_graph_tokens=False),
        closed_text_tokens=None,
    )
    special_tokens = {
        'PAD': tokenizer.pad_token_id,
        'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        'END': tokenizer.eos_token_id,
    }

    config_train = load_yaml(args.config_train)[model_name]
    if args.override_nepoch > 0:
        config_train = dict(config_train)
        config_train['nepoch'] = args.override_nepoch
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
    elif args.mode == 'optimizing':
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
        )
    else:
        model = load_model_by_mode(
            args=args,
            device=device,
            model_name=model_name,
            ntoken=ntoken,
            config_train=config_train,
            special_tokens=special_tokens,
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
    elif args.mode == 'testing':
        test_loop(
            args=args,
            dataloader=dataloader_dict[args.test_split],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            pattern_filtered=pattern_filtered,
            searching_split=args.test_split,
            resume_epoch=args.rl_resume_epoch if args.rl_resume_epoch != 0 else args.resume_epoch,
            src_len=src_len,
            tgt_len=tgt_len,
            kg=kg,
            device=device,
            accelerator=accelerator if args.accelerate else None,
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
            stage_label='before_grpo',
            kg=kg,
        )
        trainer_result = optimize_grpo(
            args=args,
            dataset=dataset_dict['train'],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            kg=kg,
            batch_size=args.batch_size,
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
            stage_label='after_grpo',
            kg=kg,
        )
        write_rl_experiment_summary(experiment_record, trainer_result)


if __name__ == '__main__':
    main()
