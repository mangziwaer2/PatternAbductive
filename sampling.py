import argparse
import json
import os
import random
from functools import partial

import pandas as pd
from tqdm import tqdm

from utils.condition import expand_sample_with_conditions
from utils.load import load_kg
from utils.textualization import attach_textual_fields


STATE_FILENAME = 'sampling_state.json'


def init_workers(init_value):
    global graph_samplers
    graph_samplers = init_value


def judge(answers_from, mode):
    if mode == 'train':
        return len(answers_from['train']) > 0
    if mode == 'valid':
        return len(answers_from['train']) > 0 and len(answers_from['valid']) > 0 \
            and len(answers_from['train']) != len(answers_from['valid'])
    if mode == 'test':
        return len(answers_from['train']) > 0 and len(answers_from['valid']) > 0 \
            and len(answers_from['test']) > 0 and len(answers_from['test']) != len(answers_from['valid'])
    return False


def sample_good_query_given_pattern(mode, max_answers_size, pattern_str):
    answers_from = {}
    while True:
        sampled_query = graph_samplers[mode].sample_valid_query_given_pattern(pattern_str)

        answers_from['train'] = graph_samplers['train'].search_answers_to_query(sampled_query)
        if mode in ['valid', 'test']:
            answers_from['valid'] = graph_samplers['valid'].search_answers_to_query(sampled_query)

        if mode == 'test':
            answers_from['test'] = graph_samplers['test'].search_answers_to_query(sampled_query)

        if len(answers_from[mode]) > max_answers_size:
            continue

        if judge(answers_from, mode):
            break

    return sampled_query, answers_from, pattern_str


def subsample_answers(answers_from, mode, max_answers_size, rng):
    sampled_answers = set(rng.sample(answers_from[mode], max_answers_size))
    sampled_answers_from = {}
    for split in ['train', 'valid', 'test']:
        if split not in answers_from:
            continue
        sampled_answers_from[split] = list(sampled_answers.intersection(answers_from[split]))
        if split == mode:
            break
    return sampled_answers_from


def build_sample_records(args, mode, answers_from, query, pattern_str, base_sample_id, rng, kg):
    if len(answers_from[mode]) > args.max_answer_size:
        while True:
            sampled_answers_from = subsample_answers(
                answers_from=answers_from,
                mode=mode,
                max_answers_size=args.max_answer_size,
                rng=rng,
            )
            if judge(sampled_answers_from, mode):
                break
        answers = sampled_answers_from[mode]
    else:
        answers = answers_from[mode]

    base_record = {
        'base_sample_id': base_sample_id,
        'answers': answers,
        'query': query,
        'pattern_str': pattern_str,
    }
    return [
        attach_textual_fields(record, kg)
        for record in expand_sample_with_conditions(
        base_record=base_record,
        samples_per_query=args.condition_samples_per_query,
        max_condition_arity=args.max_condition_arity,
        include_unconditional=args.include_unconditional,
        rng=rng,
    )]


def flush_records(records, output_path, rng):
    if not records:
        return 0
    row_count = len(records)
    rng.shuffle(records)
    with open(output_path, 'a', encoding='utf-8') as output_file:
        for record in records:
            record['answers'] = [int(idx) for idx in record['answers']]
            output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
    records.clear()
    return row_count


def prepare_output_dir(args):
    output_dir = os.path.join(args.data_root, args.dataname)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prepare_output_path(mode, args):
    output_dir = prepare_output_dir(args)
    return os.path.join(output_dir, f'{args.dataname}-{mode}-a2q.jsonl')


def prepare_state_path(args):
    return os.path.join(prepare_output_dir(args), STATE_FILENAME)


def create_initial_state(args, patterns_pool):
    return {
        'version': 1,
        'dataname': args.dataname,
        'seed': args.seed,
        'current_split': 'train',
        'splits': {
            split: {
                'next_index': 0,
                'rows_written': 0,
                'completed': False,
                'total_patterns': len(patterns_pool[split]),
            }
            for split in ['train', 'valid', 'test']
        }
    }


def save_state(args, state):
    state_path = prepare_state_path(args)
    with open(state_path, 'w', encoding='utf-8') as output_file:
        json.dump(state, output_file, ensure_ascii=False, indent=2)


def load_or_create_state(args, patterns_pool):
    state_path = prepare_state_path(args)
    if args.restart:
        if os.path.exists(state_path):
            os.remove(state_path)
        return create_initial_state(args, patterns_pool)

    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as input_file:
            state = json.load(input_file)
        for split in ['train', 'valid', 'test']:
            state['splits'].setdefault(split, {
                'next_index': 0,
                'rows_written': 0,
                'completed': False,
                'total_patterns': len(patterns_pool[split]),
            })
            state['splits'][split]['total_patterns'] = len(patterns_pool[split])
        return state

    return create_initial_state(args, patterns_pool)


def finalize_state_if_done(args, state):
    if all(state['splits'][split]['completed'] for split in ['train', 'valid', 'test']):
        state_path = prepare_state_path(args)
        if os.path.exists(state_path):
            os.remove(state_path)
        print('# Sampling completed, removed checkpoint state file.')
    else:
        save_state(args, state)


def write_stats(output_dir, kg):
    stats_path = os.path.join(output_dir, 'stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write(f'num_ent\t{kg.num_ent}\n')
        stats_file.write(f'num_rel\t{kg.num_rel}\n')


def write_text_format_manifest(output_dir, args):
    manifest_path = os.path.join(output_dir, 'text_format_manifest.json')
    manifest = {
        'format_version': 'compact_text_v1',
        'dataname': args.dataname,
        'data_root': args.data_root,
        'generated_by': 'sampling.py',
        'changes': [
            'Sampling writes compact textual fields directly.',
            'Removed entity prefix "ent:" from textual fields.',
            'Removed relation prefix "rel:" from textual fields.',
            'Kept relation direction markers "+" and "-" to preserve edge direction.',
            'Kept OBS/COND and control labels for disambiguating input structure.',
        ],
    }
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)


def sample_mode(args, mode, graph_samplers, patterns_pool, rng, kg, state):
    print(f'Sampling {mode} queries')
    output_path = prepare_output_path(mode, args)
    split_state = state['splits'][mode]
    start_index = split_state['next_index']
    if split_state['completed']:
        print(f'# Skip {mode}: already completed.')
        return
    if start_index > 0 and not os.path.exists(output_path):
        raise FileNotFoundError(
            f'Resume requested for split "{mode}" from index {start_index}, '
            f'but output file does not exist: {output_path}'
        )
    if start_index == 0 and os.path.exists(output_path):
        os.remove(output_path)
    records_buffer = []
    rows_written = split_state['rows_written']
    next_index = start_index

    init_workers(graph_samplers)
    progress = tqdm(
        range(start_index, len(patterns_pool[mode])),
        initial=start_index,
        total=len(patterns_pool[mode]),
    )
    try:
        for base_sample_id in progress:
            pattern_str = patterns_pool[mode][base_sample_id]
            func = partial(sample_good_query_given_pattern, mode, args.max_answer_size)
            sampled_query, answers_from, query_type = func(pattern_str)
            records_buffer.extend(build_sample_records(
                args=args,
                mode=mode,
                answers_from=answers_from,
                query=sampled_query,
                pattern_str=query_type,
                base_sample_id=base_sample_id,
                rng=rng,
                kg=kg,
            ))
            next_index = base_sample_id + 1
            if len(records_buffer) >= args.flush_size:
                rows_written += flush_records(records_buffer, output_path, rng)
                split_state['rows_written'] = rows_written
                split_state['next_index'] = next_index
                state['current_split'] = mode
                save_state(args, state)
            elif args.checkpoint_frequency > 0 and next_index % args.checkpoint_frequency == 0:
                rows_written += flush_records(records_buffer, output_path, rng)
                split_state['rows_written'] = rows_written
                split_state['next_index'] = next_index
                state['current_split'] = mode
                save_state(args, state)
    except KeyboardInterrupt:
        rows_written += flush_records(records_buffer, output_path, rng)
        split_state['rows_written'] = rows_written
        split_state['next_index'] = next_index
        state['current_split'] = mode
        save_state(args, state)
        print(f'# Sampling interrupted. Resume with next_index={next_index} for split={mode}.')
        raise
    finally:
        progress.close()

    if records_buffer:
        rows_written += flush_records(records_buffer, output_path, rng)

    split_state['rows_written'] = rows_written
    split_state['next_index'] = len(patterns_pool[mode])
    split_state['completed'] = True
    state['current_split'] = mode
    save_state(args, state)
    print(f'# Wrote {rows_written} rows to {output_path}')


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pattern_path', default='./metadata/pattern_filtered.csv')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--data_root', default='./sampled_data_compact/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--condition-samples-per-query', type=int, default=6)
    parser.add_argument('--max-condition-arity', type=int, default=3)
    parser.add_argument('--include-unconditional', action='store_true')
    parser.add_argument('--flush-size', type=int, default=5000)
    parser.add_argument('--checkpoint-frequency', type=int, default=1000)
    parser.add_argument('--restart', action='store_true')
    return parser.parse_args()


def main():
    args = my_parse_args()
    pattern_table = pd.read_csv(args.pattern_path, index_col='id')
    print(pattern_table)

    kg = load_kg(args.dataname)
    graph_samplers = kg.graph_samplers
    num_train_edges = kg.num_train_edges

    output_dir = prepare_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    write_stats(output_dir, kg)
    write_text_format_manifest(output_dir, args)

    print(f'# Sampling from {args.dataname} dataset, num_samples_perpattern:')
    num_samples_perpattern = {
        'train': num_train_edges,
        'valid': num_train_edges // 8,
        'test': num_train_edges // 8,
    }
    print(num_samples_perpattern)

    patterns_pool = {'train': [], 'valid': [], 'test': []}
    for _, pattern_str in pattern_table['pattern_str'].items():
        for split in ['train', 'valid', 'test']:
            patterns_pool[split].extend([pattern_str] * num_samples_perpattern[split])

    state = load_or_create_state(args, patterns_pool)
    split_offsets = {'train': 0, 'valid': 10_000, 'test': 20_000}
    for split in ['train', 'valid', 'test']:
        split_rng = random.Random(args.seed + split_offsets[split])
        split_rng.shuffle(patterns_pool[split])
        try:
            sample_mode(
                args=args,
                mode=split,
                graph_samplers=graph_samplers,
                patterns_pool=patterns_pool,
                rng=split_rng,
                kg=kg,
                state=state,
            )
        except KeyboardInterrupt:
            print('# Sampling stopped by user.')
            return

    finalize_state_if_done(args, state)


if __name__ == '__main__':
    main()
