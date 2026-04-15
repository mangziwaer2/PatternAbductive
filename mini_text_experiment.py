import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

from sampling import init_workers, sample_good_query_given_pattern, build_sample_records
from model.tokenizer import (
    create_text_tokenizer,
    decode_text_token_ids,
    get_text_extra_tokens,
    new_extract_sample_to_device_condition,
)
from model.transformer import create_transformer, GPT2_MODEL_PATH
from utils.text_constraints import TextConstraintLogitsProcessor
from utils.dataloader import new_create_dataset, REPRESENTATION_TEXT
from utils.load import load_kg, load_yaml
from utils.textualization import get_closed_text_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--data_root', default='./sampled_data_mini_exp/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-base-samples', type=int, default=18)
    parser.add_argument('--valid-base-samples', type=int, default=6)
    parser.add_argument('--test-base-samples', type=int, default=6)
    parser.add_argument('--max-answer-size', type=int, default=32)
    parser.add_argument('--condition-samples-per-query', type=int, default=4)
    parser.add_argument('--max-condition-arity', type=int, default=2)
    parser.add_argument('--include-unconditional', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--compare-samples', type=int, default=5)
    parser.add_argument('--max-gen-len', type=int, default=128)
    parser.add_argument('--use-pretrained-text-model', action='store_true')
    parser.add_argument('--constrained-generation', action='store_true')
    parser.add_argument('--log-path', default='./results/mini_text_experiment/mini_text_experiment.log')
    parser.add_argument('--compare-log-path', default='./results/mini_text_experiment/mini_text_experiment_samples.log')
    parser.add_argument('--pattern-path', default='./metadata/pattern_filtered.csv')
    parser.add_argument('--restart-data', action='store_true')
    parser.add_argument('--source-text-field', default='observation_text')
    parser.add_argument('--target-text-field', default='hypothesis_text')
    return parser.parse_args()


def write_log(message, log_path):
    safe_message = message
    try:
        print(safe_message)
    except UnicodeEncodeError:
        safe_message = safe_message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        sys.stdout.buffer.write((safe_message + '\n').encode('utf-8', errors='replace'))
        sys.stdout.flush()
    with open(log_path, 'a', encoding='utf-8') as output_file:
        output_file.write(message + '\n')


def ensure_clean_log(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)


def write_labeled_block(label, text, log_path):
    write_log(label, log_path)
    for line in str(text).splitlines():
        write_log(f'  {line}', log_path)


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def sample_split_records(mode, num_base_samples, patterns, args, kg, rng):
    init_workers(kg.graph_samplers)
    helper_args = SimpleNamespace(
        max_answer_size=args.max_answer_size,
        condition_samples_per_query=args.condition_samples_per_query,
        max_condition_arity=args.max_condition_arity,
        include_unconditional=args.include_unconditional,
    )

    records = []
    for base_sample_id in range(num_base_samples):
        pattern_str = patterns[base_sample_id % len(patterns)]
        sampled_query, answers_from, query_type = sample_good_query_given_pattern(
            mode, args.max_answer_size, pattern_str)
        records.extend(build_sample_records(
            args=helper_args,
            mode=mode,
            answers_from=answers_from,
            query=sampled_query,
            pattern_str=query_type,
            base_sample_id=base_sample_id,
            rng=rng,
            kg=kg,
        ))
    return records


def build_small_dataset(args, kg, log_path):
    dataset_dir = Path(args.data_root) / args.dataname
    if args.restart_data and dataset_dir.exists():
        for child in dataset_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil
                shutil.rmtree(child)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    pattern_table = pd.read_csv(args.pattern_path, index_col='id')
    patterns = pattern_table['pattern_str'].tolist()

    split_counts = {
        'train': args.train_base_samples,
        'valid': args.valid_base_samples,
        'test': args.test_base_samples,
    }
    split_offsets = {'train': 0, 'valid': 1000, 'test': 2000}
    split_records = {}
    for split, count in split_counts.items():
        split_rng = random.Random(args.seed + split_offsets[split])
        shuffled_patterns = patterns[:]
        split_rng.shuffle(shuffled_patterns)
        split_records[split] = sample_split_records(
            mode=split,
            num_base_samples=count,
            patterns=shuffled_patterns,
            args=args,
            kg=kg,
            rng=split_rng,
        )
        output_path = dataset_dir / f'{args.dataname}-{split}-a2q.jsonl'
        write_jsonl(output_path, split_records[split])
        write_log(f'[{split}] base_samples={count}, conditioned_rows={len(split_records[split])}', log_path)

    with open(dataset_dir / 'stats.txt', 'w', encoding='utf-8') as stats_file:
        stats_file.write(f'num_ent\t{kg.num_ent}\n')
        stats_file.write(f'num_rel\t{kg.num_rel}\n')

    return split_records


def create_small_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def prepare_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def uses_graph_text(args):
    return args.source_text_field == 'hypothesis_graph_text' or args.target_text_field == 'hypothesis_graph_text'


def build_special_tokens_from_text_tokenizer(tokenizer):
    return {
        'PAD': tokenizer.pad_token_id,
        'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        'END': tokenizer.eos_token_id,
        'SEP': tokenizer.sep_token_id,
    }


def train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, device, src_len, tgt_len):
    model.train()
    total_loss = 0.0
    steps = 0
    for sample in dataloader:
        _, _, _, input_ids, attention_mask, labels, _, _ = new_extract_sample_to_device_condition(
            device=device,
            sample=sample,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gen=False,
            condition_key='condition_text_textual',
        )
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_loss(model, tokenizer, dataloader, device, src_len, tgt_len):
    model.eval()
    total_loss = 0.0
    steps = 0
    for sample in dataloader:
        _, _, _, input_ids, attention_mask, labels, _, _ = new_extract_sample_to_device_condition(
            device=device,
            sample=sample,
            tokenizer=tokenizer,
            src_len=src_len,
            tgt_len=tgt_len,
            is_gen=False,
            condition_key='condition_text_textual',
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(outputs.loss.detach().cpu())
        steps += 1
    return total_loss / max(steps, 1)


def make_generation_input(sample, tokenizer, device, src_len):
    source = [sample['source']]
    condition_text = [sample['condition_text_textual']]
    merged_source = [source[0] if condition_text[0] == '' else f"{source[0]} SEP {condition_text[0]}"]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    tokenized = tokenizer(
        merged_source,
        padding='longest',
        max_length=src_len,
        return_tensors='pt',
    ).to(device)
    tokenizer.padding_side = original_padding_side
    return merged_source[0], tokenized


@torch.no_grad()
def generate_prediction(model, tokenizer, sample, device, src_len, max_gen_len, use_constraints, preserve_whitespace):
    model.eval()
    merged_source, tokenized = make_generation_input(sample, tokenizer, device, src_len)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=min(model.config.n_positions, input_ids.shape[-1] + max_gen_len),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=LogitsProcessorList([
            TextConstraintLogitsProcessor(
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
                condition_texts=[sample['condition_text_textual']],
            )
        ]) if use_constraints else None,
    )
    generated_tokens = output[0, input_ids.shape[-1]:]
    prediction = decode_text_token_ids(
        tokenizer,
        generated_tokens.tolist(),
        preserve_whitespace=preserve_whitespace,
    ).strip()
    return merged_source, prediction


def compare_examples(
        model,
        tokenizer,
        dataset,
        device,
        src_len,
        max_gen_len,
        num_examples,
        split_name,
        log_path,
        use_constraints,
        preserve_whitespace):
    write_log(f'===== {split_name} SAMPLE COMPARISON =====', log_path)
    sample_count = min(num_examples, len(dataset))
    for i in range(sample_count):
        sample = dataset[i]
        merged_source, prediction = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            device=device,
            src_len=src_len,
            max_gen_len=max_gen_len,
            use_constraints=use_constraints,
            preserve_whitespace=preserve_whitespace,
        )
        groundtruth = sample['target']
        write_labeled_block(f'[{split_name} #{i}] INPUT', merged_source, log_path)
        write_labeled_block(f'[{split_name} #{i}] TARGET', groundtruth, log_path)
        write_labeled_block(f'[{split_name} #{i}] PRED', prediction if prediction else '<empty>', log_path)
        write_log('', log_path)


def main():
    args = parse_args()
    ensure_clean_log(args.log_path)
    ensure_clean_log(args.compare_log_path)
    write_log('===== MINI TEXT EXPERIMENT =====', args.log_path)
    write_log(json.dumps(vars(args), ensure_ascii=False, indent=2), args.log_path)
    write_log(f'constrained_generation={args.constrained_generation}', args.log_path)
    write_log('===== MINI TEXT EXPERIMENT SAMPLE COMPARISON =====', args.compare_log_path)
    write_log(json.dumps(vars(args), ensure_ascii=False, indent=2), args.compare_log_path)

    device = prepare_device()
    write_log(f'device={device}', args.log_path)
    write_log(f'device={device}', args.compare_log_path)

    kg = load_kg(args.dataname)
    split_records = build_small_dataset(args, kg, args.log_path)

    pattern_table = pd.read_csv(args.pattern_path, index_col='id')
    dataset_splits = [split for split in ['train', 'valid', 'test'] if len(split_records.get(split, [])) > 0]
    dataset_dict, _, _ = new_create_dataset(
        dataname=args.dataname,
        pattern_filtered=pattern_table,
        data_root=args.data_root,
        splits=dataset_splits,
        kg=kg,
        source_text_field=args.source_text_field,
        target_text_field=args.target_text_field,
        representation=REPRESENTATION_TEXT,
    )
    train_dataset = dataset_dict['train']
    valid_dataset = dataset_dict['valid']
    test_dataset = dataset_dict.get('test')

    train_loader = create_small_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = create_small_dataloader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    config_dataloader = load_yaml('configs/config-dataloader.yml')
    src_len = config_dataloader['text_obs_len']
    tgt_len = config_dataloader['text_hyp_len']

    tokenizer, vocab_size = create_text_tokenizer(
        GPT2_MODEL_PATH,
        extra_tokens=get_text_extra_tokens(include_graph_tokens=uses_graph_text(args)),
        closed_text_tokens=get_closed_text_tokens(kg),
    )
    special_tokens = build_special_tokens_from_text_tokenizer(tokenizer)
    model = create_transformer(
        ntoken=vocab_size,
        special_tokens=special_tokens,
        model_name='GPT2_6_act_nt',
        vocab_size=vocab_size,
        use_pretrained_weights=args.use_pretrained_text_model,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = min(args.warmup_steps, total_steps)

    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, tokenizer, train_loader, optimizer, scheduler, device, src_len, tgt_len)
        valid_loss = evaluate_loss(model, tokenizer, valid_loader, device, src_len, tgt_len)
        history.append((epoch, train_loss, valid_loss))
        write_log(
            f'epoch={epoch:02d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}',
            args.log_path,
        )

    first_epoch, first_train_loss, first_valid_loss = history[0]
    last_epoch, last_train_loss, last_valid_loss = history[-1]
    write_log('===== LOSS SUMMARY =====', args.log_path)
    write_log(
        f'train_loss: {first_train_loss:.6f} -> {last_train_loss:.6f} '
        f'(delta={last_train_loss - first_train_loss:.6f})',
        args.log_path,
    )
    write_log(
        f'valid_loss: {first_valid_loss:.6f} -> {last_valid_loss:.6f} '
        f'(delta={last_valid_loss - first_valid_loss:.6f})',
        args.log_path,
    )
    write_log('', args.log_path)
    write_log('===== LOSS SUMMARY =====', args.compare_log_path)
    write_log(
        f'train_loss: {first_train_loss:.6f} -> {last_train_loss:.6f} '
        f'(delta={last_train_loss - first_train_loss:.6f})',
        args.compare_log_path,
    )
    write_log(
        f'valid_loss: {first_valid_loss:.6f} -> {last_valid_loss:.6f} '
        f'(delta={last_valid_loss - first_valid_loss:.6f})',
        args.compare_log_path,
    )
    write_log('', args.compare_log_path)

    preserve_whitespace = (args.target_text_field == 'hypothesis_graph_text')

    compare_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        device=device,
        src_len=src_len,
        max_gen_len=args.max_gen_len,
        num_examples=args.compare_samples,
        split_name='train',
        log_path=args.compare_log_path,
        use_constraints=args.constrained_generation,
        preserve_whitespace=preserve_whitespace,
    )
    compare_examples(
        model=model,
        tokenizer=tokenizer,
        dataset=valid_dataset,
        device=device,
        src_len=src_len,
        max_gen_len=args.max_gen_len,
        num_examples=args.compare_samples,
        split_name='valid',
        log_path=args.compare_log_path,
        use_constraints=args.constrained_generation,
        preserve_whitespace=preserve_whitespace,
    )
    if test_dataset is not None:
        compare_examples(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            device=device,
            src_len=src_len,
            max_gen_len=args.max_gen_len,
            num_examples=args.compare_samples,
            split_name='test',
            log_path=args.compare_log_path,
            use_constraints=args.constrained_generation,
            preserve_whitespace=preserve_whitespace,
        )

    write_log(f'log saved to {args.log_path}', args.log_path)
    write_log(f'compare_log saved to {args.compare_log_path}', args.log_path)
    write_log(f'compare_log saved to {args.compare_log_path}', args.compare_log_path)


if __name__ == '__main__':
    main()
