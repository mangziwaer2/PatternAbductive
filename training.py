import argparse
import csv
import datetime
import inspect
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

from model.tokenizer import (
    new_extract_sample_to_device_pattern,
    new_extract_sample_to_device,
    new_extract_sample_to_device_condition,
    create_tokenizer,
    create_text_tokenizer,
    decode_text_token_ids,
    get_text_extra_tokens,
    source_to_prompt,
)
from model.transformer import create_transformer, GPT2_MODEL_PATH
from utils.dataloader import (
    new_create_dataset,
    new_create_dataloader,
    REPRESENTATION_ID,
    REPRESENTATION_TEXT,
)
from utils.text_constraints import TextConstraintLogitsProcessor
from utils.load import load_yaml, load_kg, load_model, resolve_sampled_dataset_path
from utils.textualization import get_closed_text_tokens
from utils.text_scoring import score_text_query_batch
import logging

from utils.parsing import qry_actionstr_2_wordlist
from utils.stat_util import stat_scores_by_pattern


def load_evaluation_functions():
    from utils.evaluation import (
        scoring_input_act_batch_condition,
        scoring_input_wordlist_batch,
        scoring_input_act_batch,
    )
    return (
        scoring_input_act_batch_condition,
        scoring_input_wordlist_batch,
        scoring_input_act_batch,
    )


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
    return f'{timestamp}-{args.dataname}-{args.modelname}-{args.condition}'


def uses_graph_text(args):
    return args.source_text_field == 'hypothesis_graph_text' or args.target_text_field == 'hypothesis_graph_text'


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
            'test_split_status': 'not_used_for_experiment',
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

    initialize_csv(
        paths['loss_csv_path'],
        ['epoch', 'train_loss', 'valid_loss'],
    )
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


def write_experiment_summary(experiment_record, loss_log, args):
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
        f'- test_split_status: {experiment_record["metadata"]["data"]["test_split_status"]}',
        f'- constrained_generation: {args.constrained}',
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


def write_rl_experiment_summary(experiment_record, trainer_result, args):
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
        f'- constrained_generation: {args.constrained}',
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
    return {
        key: [value]
        for key, value in sample.items()
    }


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


def build_logged_input(source_value, condition_value):
    condition_value = format_logged_condition(condition_value).strip()
    return source_value if condition_value == '' else f'{source_value} SEP {condition_value}'


def run_generation(
        args,
        model,
        input_ids,
        attention_mask,
        max_length,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        tokenizer,
        is_gpt,
        is_constrained,
        condition_texts=None,
        do_sample=True):
    logits_processor = None
    if is_constrained and args.representation == REPRESENTATION_TEXT:
        from transformers import LogitsProcessorList
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        logits_processor = LogitsProcessorList([
            TextConstraintLogitsProcessor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                condition_texts=condition_texts,
            )
        ])
        prefix_allowed_tokens_fn = None
    elif is_constrained:
        prefix_allowed_tokens_fn = Prefix_allowed_tokens_fn(
            offset=offset,
            nentity=nentity,
            nrelation=nrelation,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
        )
    else:
        prefix_allowed_tokens_fn = None

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        min_length=-1,
        top_p=1.0,
        do_sample=do_sample,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )
    if do_sample:
        generation_kwargs['top_k'] = args.test_top_k

    return model.generate(**generation_kwargs)


@torch.no_grad()
def log_prediction_comparisons(args, dataset_dict, model, tokenizer, src_len, tgt_len,
                               is_gpt, is_act, accelerator, log_path, stage_label):
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
        emit_text_log(
            f'[{split}] logged_indices={indices}',
            log_path,
            also_print=args.comparison_console,
        )

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
                args=args,
                model=model_for_generation,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=tgt_len + src_len * (is_gpt is True),
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                is_gpt=is_gpt,
                is_constrained=(
                    args.constrained if args.representation == REPRESENTATION_TEXT
                    else is_act and args.constrained
                ),
                condition_texts=condition if args.representation == REPRESENTATION_TEXT else None,
                do_sample=False,
            )

            if is_gpt:
                mask_source(device, source_attention_mask, pred, tokenizer)

            if args.representation == REPRESENTATION_TEXT:
                prediction = decode_text_token_ids(
                    tokenizer,
                    pred[0].tolist(),
                    preserve_whitespace=(args.target_text_field == 'hypothesis_graph_text'),
                ).strip()
            else:
                prediction = tokenizer.batch_decode(pred, skip_special_tokens=True)[0].strip()

            condition_value = condition[0] if condition else ''
            emit_text_log(
                f'[{split}] idx={sample_index} pattern_id={pattern_id[0]}',
                log_path,
                also_print=args.comparison_console,
            )
            emit_text_log(
                f'[{split}] INPUT  : {build_logged_input(source[0], condition_value)}',
                log_path,
                also_print=args.comparison_console,
            )
            emit_text_log(
                f'[{split}] TARGET : {target[0]}',
                log_path,
                also_print=args.comparison_console,
            )
            emit_text_log(
                f'[{split}] PRED   : {prediction}',
                log_path,
                also_print=args.comparison_console,
            )
            emit_text_log('', log_path, also_print=args.comparison_console)


def reward_add(score: dict):
    if rl_representation == REPRESENTATION_TEXT:
        return (
            score['jaccard'] * rl_factor[0]
            + score['dice'] * rl_factor[1]
            + score['overlap'] * rl_factor[2]
            + score['condition'] * rl_factor[3]
        )
    if cond == 'relation' or cond == 'entity':
        return score['jaccard'] * rl_factor[0] + score['dice'] * rl_factor[1] + score['overlap'] * rl_factor[2] + score['spec'] * rl_factor[3]
    if cond == 'pattern':
        return score['jaccard'] * rl_factor[0] + score['dice'] * rl_factor[1] + score['overlap'] * rl_factor[2] + score['validity'] * rl_factor[3]
    if cond == 'entitynumber':
        return score['jaccard'] * rl_factor[0] + score['dice'] * rl_factor[1] + score['overlap'] * rl_factor[2] + score['enumber'] * rl_factor[3]
    return score['jaccard'] * rl_factor[0] + score['dice'] * rl_factor[1] + score['overlap'] * rl_factor[2] + score['pnumber'] * rl_factor[3]



def reward_func(prompts, completions, target, source, condition_text_textual=None, **kwargs):
    if rl_representation == REPRESENTATION_TEXT:
        if condition_text_textual is None:
            condition_text_textual = [''] * len(completions)
        scores = score_text_query_batch(
            completions=completions,
            targets=target,
            sources=source,
            condition_texts=condition_text_textual,
            kg=rl_kg,
            graph_samplers=graph_samplers,
            searching_split=rl_search_split,
        )
        return [float(reward_add(score)) for score in scores]

    _, _, scoring_input_act_batch = load_evaluation_functions()
    scores, _ = scoring_input_act_batch(
        pred_word_batch=completions,
        label_word_batch=target,
        ans_word_batch=source,
        scoring_method=rl_scoring_list,
        do_correction=do_correction,
        graph_samplers=graph_samplers,
        searching_split=rl_search_split,
        return_failures=True,
    )
    return [float(reward_add(score)) for score in scores]


def build_grpo_dataset(dataset, args):
    dataset = dataset.map(lambda example: source_to_prompt(example, args=args))
    keep_columns = ['prompt', 'source', 'target', 'condition_text_textual']
    removable = [column for column in dataset.column_names if column not in keep_columns]
    if removable:
        dataset = dataset.remove_columns(removable)
    return dataset


def optimize_gpro(args, dataset, model, tokenizer, graph_sampler, kg, batch_size,
                  is_gpt, is_act, src_len, tgt_len, experiment_record=None):
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError('TRL is required for optimizing mode. Please install `trl`.') from exc
    print('GRPO Setting Up')
    dataset = build_grpo_dataset(dataset, args)
    output_dir = experiment_record['paths']['experiment_dir'] if experiment_record is not None else f'./results/optim/{build_experiment_name(args)}'
    report_to = None if str(args.rl_report_to).lower() in {'', 'none', 'null'} else args.rl_report_to
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

    global graph_samplers
    graph_samplers = graph_sampler
    global rl_kg
    rl_kg = kg
    global rl_search_split
    rl_search_split = args.rl_search_split
    global do_correction
    do_correction = args.do_correction
    global rl_scoring_list
    if args.condition == 'relation' or args.condition == 'entity':
        rl_scoring_list = ['jaccard', 'dice', 'overlap', 'specific']
    else:
        rl_scoring_list = ['jaccard', 'dice', 'overlap', 'validity']
    global cond
    cond = args.condition
    global rl_representation
    rl_representation = args.representation
    global rl_factor
    parsed_factors = list(eval(args.rl_factor))
    while len(parsed_factors) < 4:
        parsed_factors.append(1.0)
    rl_factor = parsed_factors[:4]
    print(rl_factor)

    original_forward = None
    if 'logits_to_keep' not in inspect.signature(model.forward).parameters:
        original_forward = model.forward

        def forward_with_grpo_compat(*forward_args, **forward_kwargs):
            forward_kwargs.pop('logits_to_keep', None)
            return original_forward(*forward_args, **forward_kwargs)

        model.forward = forward_with_grpo_compat

    model.warnings_issued = {}

    def dummy_add_model_tags(self, tags):
        pass

    model.add_model_tags = dummy_add_model_tags.__get__(model)
    trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    trainer_result = trainer.train()
    if original_forward is not None:
        model.forward = original_forward
    trainer.save_model(output_dir)
    ckpt_path = os.path.join(args.checkpoint_root, args.modelname, \
                             f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.rl_epochs}-optimize-{args.condition}.pth')
    save_model(ckpt_path, 'model', model, epoch=args.rl_epochs)
    if experiment_record is not None:
        append_text_log(experiment_record['paths']['run_log_path'], json.dumps(trainer_result.metrics, ensure_ascii=False, indent=2))
    return trainer_result

def extract_sample_batch(args, device, sample, tokenizer, src_len, tgt_len, is_gen: bool):
    if args.condition == 'unconditional':
        return new_extract_sample_to_device(
            device, sample, tokenizer, src_len, tgt_len, is_gen)
    if args.condition == 'pattern-legacy':
        return new_extract_sample_to_device_pattern(
            device, sample, tokenizer, src_len, tgt_len, is_gen)
    return new_extract_sample_to_device_condition(
        device, sample, tokenizer, src_len, tgt_len, is_gen,
        condition_key=args.condition_field)


def train_loop(args, model, tokenizer, optimizer, scheduler, dataloader,
               device, src_len, tgt_len, accelerator=None, epoch=None, total_epochs=None,
               on_log_step=None):
    model.train()
    niter = len(dataloader)
    total_loss = 0.0
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
        batch = extract_sample_batch(
            args=args,
            device=device,
            sample=sample,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gen=False,
        )
        source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, _ = batch

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss_value = float(loss.detach().item())
        total_loss += loss_value
        window_losses.append(loss_value)

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        if should_emit_periodic_log(step, effective_total, args.train_log_every):
            elapsed = max(time.time() - start_time, 1e-8)
            avg_loss = total_loss / step
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

    denom = effective_total
    return total_loss / max(denom, 1)


@torch.no_grad()
def evaluate_loop(args, model, tokenizer, dataloader, device, src_len, tgt_len, accelerator=None, max_batches=None):
    model.eval()
    niter = len(dataloader)
    total_loss = 0.0
    total_steps = 0

    for iter, sample in enumerate(dataloader):
        batch = extract_sample_batch(
            args=args,
            device=device,
            sample=sample,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gen=False,
        )
        _, _, _, input_ids, attention_mask, labels, _, _ = batch

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.detach()

        if accelerator is not None:
            gathered_loss = accelerator.gather(loss.reshape(1))
            total_loss += gathered_loss.float().mean().item()
        else:
            total_loss += float(loss.cpu())

        total_steps += 1
        effective_max_batches = max_batches if max_batches is not None else args.max_valid_batches
        if effective_max_batches > 0 and (iter + 1) >= effective_max_batches:
            break

    effective_max_batches = max_batches if max_batches is not None else args.max_valid_batches
    denom = min(niter, effective_max_batches) if effective_max_batches > 0 else niter
    return total_loss / max(min(total_steps, denom), 1)


def load_model_by_mode(args, device, model_name, is_gpt, ntoken=None, config_train=None):

    if args.mode in ['training', 'testing', 'optimizing'] and args.resume_epoch != 0:
        if args.tuning:
            resume_path = os.path.join(args.checkpoint_root, args.modelname, \
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}-{args.condition}.pth')
        else:
            resume_path = os.path.join(args.checkpoint_root, args.modelname, \
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}-unconditional.pth')

        print(f'Loading model: {resume_path}')
        model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'model', return_huggingface_model=True, epoch=args.resume_epoch)
        # last_epoch=0
        model.to(device)
        # Overwrite model name
        model.model_name = model_name

    if args.mode in ['training', 'optimizing'] and args.resume_epoch == 0:
        print('Creating model')
        model = create_transformer(
            ntoken=ntoken,
            special_tokens=special_tokens,
            model_name=model_name,
            vocab_size=ntoken if args.representation == REPRESENTATION_TEXT else None,
            use_pretrained_weights=(args.representation == REPRESENTATION_TEXT and args.use_pretrained_text_model),
        ).to(device)
        optimizer = None
        scheduler = None
        if args.mode == 'training':
            optimizer = torch.optim.Adam(model.parameters(),
                lr=float(config_train["lr"]))
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                start_factor=0.1, total_iters=config_train["warm_up"])
        last_epoch=0
        loss_log = {'train': {}, 'valid': {}}

    if args.mode in ['optimizing', 'testing'] and args.rl_resume_epoch != 0: # Load TRL model wrapper directly
        resume_path = os.path.join(args.checkpoint_root, args.modelname, \
            f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.rl_resume_epoch}-optimize-{args.condition}.pth')
        print(f'Loading model: {resume_path}')
        if args.rl_type=='GRPO':
            model, optimizer, scheduler, last_epoch, loss_log = \
            load_model(resume_path, 'rlmodel', return_huggingface_model=True, epoch=args.resume_epoch)

        model.model_name = model_name
        model.to(device)

    if args.mode == 'optimizing' and args.rl_resume_epoch == 0: # Convert to TRL wrapper.
        if args.rl_use_peft:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=4 if is_gpt else 8, # 4 to 16 for GPT-2 as suggested in the LoRA Paper
                lora_alpha=32 if is_gpt else 8,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM" if is_gpt else "SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, peft_config=lora_config)

    print('model.config:')
    print(model.config)

    if args.mode == 'training': return model, optimizer, scheduler, last_epoch, loss_log
    else: return model

def fit(args, nepoch, dataloader, model, tokenizer, optimizer, scheduler, graph_samplers,
        model_name, is_gpt, is_act, src_len, tgt_len,
        last_epoch, loss_log, accelerator, dataset_dict, experiment_record=None):
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
    result_path = os.path.join(args.result_root, args.modelname,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}_results.txt')
    pathlib.Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(last_epoch+1, nepoch+1): # epoch starts from 1
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
                    f'step {step}/{effective_total} '
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
                and step % args.intra_epoch_comparison_every == 0
            ):
                log_prediction_comparisons(
                    args=args,
                    dataset_dict=dataset_dict,
                    model=model,
                    tokenizer=tokenizer,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    is_gpt=is_gpt,
                    is_act=is_act,
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
            on_log_step=on_train_log_step)

        if ('-5' in model_name or '-01' in model_name):
            scheduler.step()
        # exit()
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
        with open(result_path, 'a') as result_file:
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
                    is_gpt=is_gpt,
                    is_act=is_act,
                    accelerator=accelerator,
                    log_path=experiment_record['paths']['comparison_log_path'],
                    stage_label=f'epoch_{epoch}',
                )

        # Saving checkpoint
        if epoch % args.save_frequency == 0 \
            or epoch == nepoch:
            ckpt_path = os.path.join(args.checkpoint_root, args.modelname,\
                f'{args.dataname}-{args.scale}-{args.max_answer_size}-{epoch}-{args.condition}.pth')
            save_model(ckpt_path, 'model', model, optimizer, scheduler, epoch, loss_log)

        print('=' * 50)

    if experiment_record is not None:
        log_prediction_comparisons(
            args=args,
            dataset_dict=dataset_dict,
            model=model,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gpt=is_gpt,
            is_act=is_act,
            accelerator=accelerator,
            log_path=experiment_record['paths']['comparison_log_path'],
            stage_label='final',
        )
        write_experiment_summary(
            experiment_record=experiment_record,
            loss_log=loss_log,
            args=args,
        )

def rl_suffix_name(args, iter):
    name = f'ppo_{args.ppo_lr}'\
            + f'_{args.ppo_smatch_factor}'\
            + f'_{args.ppo_init_kl_coef}'\
            + f'_{args.ppo_cliprange}'\
            + f'_{args.ppo_minibatch}'\
            + f'_{args.ppo_horizon}'\
            + (f'_{args.ppo_epochs}' if args.ppo_epochs != 4 else '')\
            + (f'_{args.ppo_share_embed_layer}' if args.ppo_share_embed_layer else '')\
            + ('_nodecay' if args.ppo_lr_no_decay else '')\
            + ('_peft' if args.ppo_use_peft else '')\
            + (f'_s{args.ppo_search_split}' if args.ppo_search_split != 'train' else '')\
            + f'x{args.ppo_proportion}'\
            + f'-{iter}'
    return name

def mask_source(device, source_attention_mask, pred, tokenizer):
    # print('source mask')
    # print(source_attention_mask[:3, :15])
    B = pred.shape[0]
    diff = pred.shape[-1] - source_attention_mask.shape[-1]
    prefix_mask = torch.cat([
        source_attention_mask,
        torch.zeros((B, diff), dtype=torch.bool, device=device)], dim=1).to(device)
    # print('prefix mask')
    # print(prefix_mask[:3, :15])
    pred[prefix_mask == 1] = tokenizer.pad_token_id

def qry_actionprefix_get_branching(action_prefix: list):
    stack = qry_actionstr_2_wordlist(
        actionstr=action_prefix,
        return_stack=True)
    return 'EMPTY' if not stack else stack[-1]

class Prefix_allowed_tokens_fn:
    def __init__(self, offset, nentity, nrelation, special_tokens, tokenizer):
        self.offset = offset
        self.nentity = nentity
        self.nrelation = nrelation
        self.special_tokens = special_tokens
        # self.bos_token_id = tokenizer.bos_token_id
        # self.sep_token_id = tokenizer.sep_token_id
        # self.eos_token_id = tokenizer.eos_token_id
        # self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.iun_ids = tokenizer.convert_tokens_to_ids(['i', 'u', 'n'])
    def get_gathered_tokens(self) -> list:
        return list(range(self.offset + self.nentity + self.nrelation))
    def get_non_special_tokens(self) -> list:
        return self.iun_ids + list(range(self.offset, self.offset + self.nentity + self.nrelation))
    # def get_p_allowed_tokens(offset:int, nentity:int, nrelation:int, rel: int) -> list:
    #     # return list(range(offset, offset + nentity))
    #     return [15,16,17]\
    #         + ([offset + h for h in rel_2_allowed_headent[rel]] if rel in rel_2_allowed_headent \
    #             else list(range(offset, offset + nentity)))\
    #         + list(range(offset + nentity, offset + nentity + nrelation))
    def get_iun_allowed_tokens(self) -> list:
        # ANY \ e
        return self.iun_ids\
                + list(range(self.offset + self.nentity, self.offset + self.nentity + self.nrelation))
    def __call__(self, batch_id: int, input_ids: torch.LongTensor) -> list:
        #初始可以随便
        if input_ids.shape[-1] <= 1:
            return self.get_gathered_tokens()
        is_gpt = (not input_ids[1] in self.get_iun_allowed_tokens())
        prefix_ids = list(input_ids)
        if is_gpt:
            #取后面的部分
            if self.tokenizer.sep_token_id in prefix_ids:
                sep_pos = prefix_ids.index(self.tokenizer.sep_token_id)
                prefix_ids = prefix_ids[sep_pos:]
            else: # Query part does not appear
                return self.get_gathered_tokens()
        # print('prefix')
        # print(prefix_ids)
        #取当前最后一个token
        last_action = prefix_ids[-1]
        #相当于是第一个tgt的生成
        if last_action in [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id]:
            return self.get_non_special_tokens()
        #如果是i,n,u则考虑i，n，u和p
        elif last_action in self.iun_ids: # ANY \ e
            return self.get_iun_allowed_tokens()
        #如果是e，检查是否成功。如果不成功就继续生成。
        elif last_action >= offset and last_action < offset + nentity:
            # omit the first 'START' token
            actionstr_prefix = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            branching = qry_actionprefix_get_branching(action_prefix=actionstr_prefix)
            if branching == 'EMPTY': # Query graph is complete, must return EOS
                return [self.tokenizer.eos_token_id]
            else: # i / u
                return self.get_iun_allowed_tokens()
        # elif operator == 'p': # ANY with constraints
        #如果是p，后面就是p或e
        elif last_action >= offset + nentity:
            return self.get_non_special_tokens()
        else: # eos, pad, etc.
            return [self.tokenizer.pad_token_id]

def constrained_inference(args, model, input_ids, attention_mask, max_length,
              bos_token_id, eos_token_id, pad_token_id, tokenizer,
              is_gpt, is_constrained, condition_texts=None):
    """
    Reference:
    https://github.com/huggingface/transformers/blob/31ec2cb2badfbdd4c1ac9c6c9b8a74e974984206/src/transformers/generation_utils.py#L1622
    """
    # num_beams = 4
    return run_generation(
        args=args,
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
        is_gpt=is_gpt,
        is_constrained=is_constrained,
        condition_texts=condition_texts,
        do_sample=True,
    )


def test_loop(args, dataloader, model, tokenizer, graph_samplers, searching_split, resume_epoch,
            is_gpt, is_act, src_len, tgt_len, kg,
            accelerator,
            score_file_suffix='test'):
    scoring_input_act_batch_condition, scoring_input_wordlist_batch, _ = load_evaluation_functions()
    score_file_suffix = f'test|{args.test_proportion}x{args.test_split}_topk{args.test_top_k}_{args.constrained}_{args.test_count0}'
    if args.rl_resume_epoch != 0:
        score_file_suffix += f'|{rl_suffix_name(args, args.rl_resume_epoch)}'

    if not accelerator is None:
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    model.eval()
    niter = len(dataloader)
    scores_all = []
    pattern_id_all = []

    import torch.distributed as dist
    with torch.no_grad():
        for iter, sample in (pbar := tqdm(enumerate(dataloader, start=1),
                                          total=niter, disable=(accelerator is not None) and (not accelerator.is_local_main_process))):
            # gathered_sample = accelerator.gather_for_metrics(sample) if accelerator is not None else sample
            source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, condition = \
                extract_sample_batch(
                    args=args,
                    device=device,
                    sample=sample,
                    tokenizer=tokenizer,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    is_gen=True,
                )

            pred = constrained_inference(args,
                model if accelerator is None else accelerator.unwrap_model(model),
                input_ids, attention_mask,
                max_length=tgt_len + src_len * (is_gpt == True),
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                is_gpt=is_gpt,
                is_constrained=(args.constrained if args.representation == REPRESENTATION_TEXT else is_act and args.constrained),
                condition_texts=condition if args.representation == REPRESENTATION_TEXT else None)

            if is_gpt: mask_source(device, source_attention_mask, pred, tokenizer)
            if args.representation == REPRESENTATION_TEXT:
                pred_decoded = [decode_text_token_ids(tokenizer, sequence.tolist()) for sequence in pred]
            else:
                pred_decoded = tokenizer.batch_decode(pred, skip_special_tokens=True)

            print('source')
            print(source[:5])
            print('target (label)')
            print(target[:5])
            print('pred_de')
            print(pred_decoded[:5])

            if args.condition ==  'relation' or args.condition == 'entity':
                scoring_method=['smatch', 'precrecf1', 'jaccard','dice','overlap','tanimoto','validity','specific'] + ['count0'] * (args.test_count0 == True)
            else:
                scoring_method=['smatch', 'precrecf1', 'jaccard','dice','overlap','tanimoto','validity'] + ['count0'] * (args.test_count0 == True)
            if args.representation == REPRESENTATION_TEXT:
                scores = score_text_query_batch(
                    completions=pred_decoded,
                    targets=target,
                    sources=source,
                    condition_texts=condition,
                    kg=kg,
                    graph_samplers=graph_samplers,
                    searching_split=searching_split,
                )
                failures_batch_id = []
            else:
                if is_act:
                    scores, failures_batch_id = scoring_input_act_batch_condition(
                        pred_word_batch=pred_decoded,
                        label_word_batch=target,
                        ans_word_batch=source,
                        condition_batch=condition,
                        scoring_method=scoring_method,
                        do_correction=args.do_correction,
                        graph_samplers=graph_samplers,
                        searching_split=searching_split,
                        return_failures=True,
                        verbose=args.vs)
                else:
                    scores, failures_batch_id = scoring_input_wordlist_batch(
                        pred_word_batch=pred_decoded,
                        label_word_batch=target,
                        ans_word_batch=source,
                        scoring_method=scoring_method,
                        do_correction=args.do_correction,
                        graph_samplers=graph_samplers,
                        searching_split=searching_split,
                        return_failures=True,
                        verbose=args.vs)
            # print(scores)
            if accelerator is not None:
                gathered_scores = [None] * accelerator.num_processes
                dist.all_gather_object(gathered_scores, scores)
                gathered_scores = [s for l in gathered_scores for s in l ]
                gathered_pattern_id = accelerator.gather(pattern_id)
            else:
                gathered_scores = scores
                gathered_pattern_id = pattern_id

            if (accelerator is None) or (accelerator.is_main_process):
                scores_all.extend(gathered_scores)
                pattern_id_all.extend(gathered_pattern_id)
                score_df = stat_scores_by_pattern(scores_all, pattern_id_all, pattern_filtered)
                # print(score_df)
                pbar.set_description(f's: {round(score_df.loc["all",("smatch","mean")], 4)}, j: {round(score_df.loc["all",("jaccard","mean")], 4)}')
                scores_path = os.path.join(args.result_root, args.modelname,\
                    f'{args.dataname}-{args.scale}-{args.max_answer_size}-{resume_epoch}-scores({score_file_suffix}).csv')

                score_df.to_csv(scores_path)

    return score_df

def save_model(path, contents:str,
               model, optimizer=None, scheduler=None, epoch=None, loss_log=None):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if contents == 'state_dicts':
        print(f'# Saving checkpoint (state_dicts) {path}')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss_log': loss_log
        }, path)
    elif contents == 'model':
        print(f'# Saving checkpoint (model) {path}')
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': epoch,
            'loss_log': loss_log
        }, path)
    else:
        print(f'# Error: contents "{contents}" not supported')
        exit()

def my_parse_args():
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--modelname', default="GPT2_6_act_nt")
    parser.add_argument('--config-dataloader', default='configs/config-dataloader.yml')
    parser.add_argument('--config-train', default='configs/config-train.yml')
    parser.add_argument('--batch_size', default=8,type=int)
    parser.add_argument('--seed', type=int, default=42)

    # Data
    parser.add_argument('--data_root', default='./sampled_data_compact/')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('--scale', default='default')
    # condition
    parser.add_argument('--condition', default='conditioned')
    parser.add_argument('--condition_field', default='condition_text')
    parser.add_argument('--representation', default=REPRESENTATION_TEXT, choices=[REPRESENTATION_ID, REPRESENTATION_TEXT])
    parser.add_argument('--source_text_field', default='observation_text')
    parser.add_argument('--target_text_field', default='hypothesis_text')
    parser.add_argument('--use_pretrained_text_model', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    # Checkpoint
    parser.add_argument('--checkpoint_root', default='./ckpt/')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)

    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')
    parser.add_argument('--do_correction', action='store_true', help='verbose flag for smatch result')

    # Testing
    parser.add_argument('--test_proportion', type=float, default=1)
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--test_top_k', type=int, default=0)
    parser.add_argument('--test_count0', action='store_true')
    parser.add_argument('--result_root', default='./results/')

    parser.add_argument('--save_frequency', type=int, default=1)

    # rl
    parser.add_argument('--rl_type', default='GRPO')
    parser.add_argument('--rl_resume_epoch', default=0)
    parser.add_argument('--rl_proportion', type=float, default=1)
    parser.add_argument('--rl_smatch_factor', type=float, default=0)
    parser.add_argument('--rl_init_kl_coef', type=float, default=0.2)
    parser.add_argument('--rl_cliprange', type=float, default=0.2)
    parser.add_argument('--rl_minibatch', type=int, default=1)
    parser.add_argument('--rl_horizon', type=int, default=10000)
    parser.add_argument('--rl_lr', type=float, default=1e-6)
    parser.add_argument('--rl_epochs', type=int, default=4)
    parser.add_argument('--rl_search_split', default='train')
    parser.add_argument('--rl_share_embed_layer', action='store_true')
    parser.add_argument('--rl_lr_no_decay', action='store_true')
    parser.add_argument('--rl_use_peft', action='store_true')
    parser.add_argument('--rl_top_k', default=0.0)
    parser.add_argument('--rl_factor', type=str, default='[1.0, 1.0, 1.0, 1.0]')
    parser.add_argument('--rl_max_steps', type=int, default=-1)
    parser.add_argument('--rl_num_generations', type=int, default=4)
    parser.add_argument('--rl_max_prompt_length', type=int, default=128)
    parser.add_argument('--rl_max_completion_length', type=int, default=128)
    parser.add_argument('--rl_logging_steps', type=int, default=10)
    parser.add_argument('--rl_save_steps', type=int, default=100)
    parser.add_argument('--rl_log_completions', action='store_true')
    parser.add_argument('--rl_report_to', default='none')

    parser.add_argument('--mode', default="training")
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--constrained', type=str2bool, default=False)
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

    parser.add_argument('--pattern_path', type=str, default="./metadata/pattern_filtered.csv")

    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()
    print(f'args:\n{args}\n')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(os.path.join(args.result_root, args.modelname)):
        os.makedirs(os.path.join(args.result_root, args.modelname))

    global config_dataloader
    config_dataloader = load_yaml(args.config_dataloader)

    global offset, special_tokens
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    if args.representation == REPRESENTATION_TEXT and args.condition_field == 'condition_text':
        args.condition_field = 'condition_text_textual'
    print(f'config_dataloader:\n{config_dataloader}\n')

    global pattern_filtered
    pattern_filtered = pd.read_csv(args.pattern_path, index_col='id')

    # Graphs (for evaluation)
    print('Loading graph')
    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers

    # Device
    global device
    if args.accelerate and args.mode != 'optimizing':
        if Accelerator is None:
            raise ImportError('accelerate is not installed. Please install it or run without --accelerate.')
        accelerator = Accelerator()
        device = accelerator.device
    else:
        accelerator = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # Model information
    model_name = args.modelname
    is_gpt = ('GPT2' in model_name)
    is_act = ('act' in model_name)

    if args.representation == REPRESENTATION_TEXT:
        src_len = config_dataloader['text_obs_len']
        tgt_len = config_dataloader['text_hyp_len']
        is_act = False
    else:
        tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
        src_len = config_dataloader['ans_len'] + 1
    print(f'model_name:{model_name}\n')

    # Batch size
    batch_size = args.batch_size
    print(f'batch_size:{batch_size}\n')

    print('=' * 50)

    # Dataset
    if args.mode == 'training':
        splits = ['train']
        try:
            resolve_sampled_dataset_path(args.data_root, args.dataname, 'valid')
            splits.append('valid')
        except FileNotFoundError:
            print('# Warning: valid split not found yet, training will use train split only.')
    elif args.mode == 'testing':
        splits = [args.test_split]
    elif args.mode == 'optimizing':
        splits = ['train']
        try:
            resolve_sampled_dataset_path(args.data_root, args.dataname, 'valid')
            splits.append('valid')
        except FileNotFoundError:
            pass
        if args.rl_search_split not in splits:
            splits.append(args.rl_search_split)
    elif args.mode == 'load-save-test':
        splits = ['train', 'test']

    print('Creating dataset & dataloader')
    global nentity, nrelation
    dataset_dict, nentity, nrelation = new_create_dataset(
        dataname=args.dataname,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=splits,
        max_rows_by_split={
            'train': args.max_train_rows,
            'valid': args.max_valid_rows,
            args.test_split: (
                args.max_valid_rows if args.mode == 'testing' else 0
            ),
        },
        kg=kg,
        source_text_field=args.source_text_field,
        target_text_field=args.target_text_field,
        representation=args.representation,
        dataset_cache_root=args.dataset_cache_root,
        dataset_num_proc=args.dataset_num_proc,
        dataset_map_batch_size=args.dataset_map_batch_size,
    )

    if args.max_train_rows > 0 and 'train' in dataset_dict:
        nrows = min(dataset_dict['train'].shape[0], args.max_train_rows)
        dataset_dict['train'] = dataset_dict['train'].select(range(nrows))
    if args.max_valid_rows > 0 and 'valid' in dataset_dict:
        nrows = min(dataset_dict['valid'].shape[0], args.max_valid_rows)
        dataset_dict['valid'] = dataset_dict['valid'].select(range(nrows))

    if args.mode == 'testing' and args.test_proportion < 1:
        nrows = dataset_dict[args.test_split].shape[0]
        dataset_dict[args.test_split] = dataset_dict[args.test_split].select(
            random.sample(range(nrows), int(nrows * args.test_proportion)))
    if args.mode == 'optimizing' and args.rl_proportion < 1:
        nrows = dataset_dict['train'].shape[0]
        dataset_dict['train'] = dataset_dict['train'].select(
            random.sample(range(nrows), int(nrows * args.rl_proportion)))
    dataloader_dict = new_create_dataloader(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        drop_last=(args.mode == 'optimizing')  # or (args.mode == 'testing' and args.accelerate)
    )

    # Tokenizer
    print('Creating tokenizer')
    if args.representation == REPRESENTATION_TEXT:
        tokenizer, ntoken = create_text_tokenizer(
            GPT2_MODEL_PATH,
            extra_tokens=get_text_extra_tokens(include_graph_tokens=uses_graph_text(args)),
            closed_text_tokens=get_closed_text_tokens(kg),
        )
        special_tokens = {
            'PAD': tokenizer.pad_token_id,
            'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
            'END': tokenizer.eos_token_id,
            'SEP': tokenizer.sep_token_id,
        }
    else:
        tokenizer, ntoken = create_tokenizer(
            special_tokens=special_tokens,
            offset=offset,
            nentity=nentity,
            nrelation=nrelation,
        )

    config_train = load_yaml(args.config_train)
    config_train = config_train[model_name]
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
            args=args, device=device, model_name=model_name, is_gpt=is_gpt, ntoken=ntoken, config_train=config_train)
    else:
        model = load_model_by_mode(
            args=args, device=device, model_name=model_name, is_gpt=is_gpt, ntoken=ntoken, config_train=config_train)

    if args.mode == 'training':
        nepoch = config_train['nepoch']
        fit(args, nepoch, dataloader_dict, model,
            tokenizer, optimizer, scheduler, graph_samplers,
            model_name, is_gpt, is_act, src_len, tgt_len,
            last_epoch, loss_log,
            accelerator=accelerator if args.accelerate else None,
            dataset_dict=dataset_dict,
            experiment_record=experiment_record)
    elif args.mode == 'testing':
        # preprocess_allowed_rel_ent_map(graph_samplers)
        test_loop(
            args=args,
            dataloader=dataloader_dict[args.test_split],
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            searching_split=args.test_split,
            resume_epoch=args.resume_epoch,
            is_gpt=is_gpt, is_act=is_act,
            src_len=src_len, tgt_len=tgt_len, kg=kg,
            accelerator=accelerator if args.accelerate else None)
    elif args.mode == 'optimizing':
        if args.rl_type == 'GRPO':
            log_prediction_comparisons(
                args=args,
                dataset_dict=dataset_dict,
                model=model,
                tokenizer=tokenizer,
                src_len=src_len,
                tgt_len=tgt_len,
                is_gpt=is_gpt,
                is_act=is_act,
                accelerator=None,
                log_path=experiment_record['paths']['comparison_log_path'],
                stage_label='before_grpo',
            )
            trainer_result = optimize_gpro(
                args=args,
                dataset=dataset_dict['train'],
                model=model,
                tokenizer=tokenizer,
                graph_sampler=graph_samplers,
                kg=kg,
                batch_size=batch_size,
                is_gpt=is_gpt, is_act=is_act,
                src_len=src_len, tgt_len=tgt_len,
                experiment_record=experiment_record,
            )
            log_prediction_comparisons(
                args=args,
                dataset_dict=dataset_dict,
                model=model,
                tokenizer=tokenizer,
                src_len=src_len,
                tgt_len=tgt_len,
                is_gpt=is_gpt,
                is_act=is_act,
                accelerator=None,
                log_path=experiment_record['paths']['comparison_log_path'],
                stage_label='after_grpo',
            )
            write_rl_experiment_summary(experiment_record, trainer_result, args)
        else:
            print('ppo is writing now')

if __name__ == '__main__':
    main()
