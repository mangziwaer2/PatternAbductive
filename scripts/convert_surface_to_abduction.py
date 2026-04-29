import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.load import load_kg, resolve_sampled_dataset_path
from utils.logic_dsl import surface_query_to_dsl
from utils.action_supervision import extract_observation_entity_tokens
from utils.text_dataset import build_stage2_trace


STATELESS_FORMAT_VERSION = 'abduction_sft_v1'


class BoundedCache:
    def __init__(self, max_items: int):
        self.max_items = max(0, int(max_items))
        self.data = OrderedDict()

    def get(self, key):
        if self.max_items <= 0 or key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key, value):
        if self.max_items <= 0:
            return
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        while len(self.data) > self.max_items:
            self.data.popitem(last=False)


def resolve_output_path(output_root, dataname, split):
    output_dir = os.path.join(output_root, dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(output_dir, f'{dataname}-{split}-a2q.jsonl')


def write_stats(output_root, dataname, kg):
    output_dir = os.path.join(output_root, dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stats_path = os.path.join(output_dir, 'stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write(f'num_ent\t{kg.num_ent}\n')
        stats_file.write(f'num_rel\t{kg.num_rel}\n')


def write_manifest(args, split_stats):
    output_dir = os.path.join(args.output_root, args.dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {
        'format_version': STATELESS_FORMAT_VERSION,
        'generated_by': 'scripts/convert_surface_to_abduction.py',
        'input_root': args.input_root,
        'output_root': args.output_root,
        'dataname': args.dataname,
        'splits': args.splits,
        'dedupe': args.dedupe,
        'result_top_k': args.result_top_k,
        'max_observation_entities': args.max_observation_entities,
        'trace_mode': args.trace_mode,
        'trace_cache_size': args.trace_cache_size,
        'max_rows': args.max_rows,
        'split_stats': split_stats,
        'fields': ['pattern_str', 'observation_text', 'logic_dsl', 'stage2_trace'],
        'notes': [
            'condition_text, condition_signature, kg_hints_text, and hypothesis_text are intentionally dropped.',
            'logic_dsl is derived from hypothesis_text.',
            'stage2_trace is generated eagerly when trace_mode=eager; otherwise dataloader derives it lazily.',
            'stage2_trace uses graph-search ACTION/RESULT events: FIND_COMMON/FIND_ALTERNATIVE/FIND_EXCLUSION, EXPAND, CHECK_COVERAGE.',
            'dataloader derives Stage 1 and Stage 2 training samples from these compact fields.',
        ],
    }
    manifest_path = os.path.join(output_dir, 'text_format_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)


def make_dedupe_key(record):
    text = '\n'.join([
        str(record.get('pattern_str', '')),
        str(record.get('observation_text', '')),
        str(record.get('hypothesis_text', record.get('logic_dsl', ''))),
    ])
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def build_compact_record(record, kg, graph_split, result_top_k, trace_mode, trace_cache, dsl_cache):
    pattern_str = record['pattern_str']
    observation_text = record['observation_text']
    if 'logic_dsl' in record and str(record['logic_dsl']).strip():
        logic_dsl = str(record['logic_dsl']).strip()
    else:
        hypothesis_text = record['hypothesis_text']
        cached_dsl = dsl_cache.get(hypothesis_text)
        if cached_dsl is None:
            cached_dsl = surface_query_to_dsl(hypothesis_text, kg)
            dsl_cache.put(hypothesis_text, cached_dsl)
        logic_dsl = cached_dsl

    if trace_mode == 'lazy':
        stage2_trace = []
    else:
        trace_key = (graph_split, pattern_str, observation_text, int(result_top_k))
        stage2_trace = trace_cache.get(trace_key)
        if stage2_trace is None:
            stage2_trace = build_stage2_trace(
                pattern_str=pattern_str,
                observation_text=observation_text,
                kg=kg,
                graph_split=graph_split,
                top_k=result_top_k,
            )
            trace_cache.put(trace_key, stage2_trace)

    return {
        'pattern_str': pattern_str,
        'observation_text': observation_text,
        'logic_dsl': logic_dsl,
        'stage2_trace': stage2_trace,
    }


def cap_observation_text(observation_text: str, max_entities: int) -> str:
    if max_entities is None or max_entities <= 0:
        return observation_text
    targets = extract_observation_entity_tokens(observation_text)
    if len(targets) <= max_entities:
        return observation_text
    return 'OBS ' + ' '.join(targets[:max_entities])


def convert_split(args, kg, split):
    input_path = resolve_sampled_dataset_path(args.input_root, args.dataname, split)
    output_path = resolve_output_path(args.output_root, args.dataname, split)
    tmp_output_path = output_path + '.tmp'
    error_path = output_path + '.errors.jsonl'
    tmp_error_path = error_path + '.tmp'
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f'Output exists: {output_path}. Use --overwrite to replace it.')
    if args.overwrite:
        for path in [output_path, tmp_output_path, error_path, tmp_error_path]:
            if os.path.exists(path):
                os.remove(path)

    seen = set()
    trace_cache = BoundedCache(args.trace_cache_size)
    dsl_cache = BoundedCache(args.dsl_cache_size)
    stats = {
        'read_rows': 0,
        'written_rows': 0,
        'duplicate_rows': 0,
        'error_rows': 0,
        'input_path': input_path,
        'output_path': output_path,
    }

    with open(input_path, 'r', encoding='utf-8') as input_file, \
            open(tmp_output_path, 'w', encoding='utf-8') as output_file, \
            open(tmp_error_path, 'w', encoding='utf-8') as error_file:
        progress = tqdm(input_file, desc=f'convert_{split}', unit='row')
        for line in progress:
            if args.max_rows > 0 and stats['read_rows'] >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            stats['read_rows'] += 1
            try:
                record = json.loads(line)
                if args.max_observation_entities > 0 and 'observation_text' in record:
                    record = dict(record)
                    record['observation_text'] = cap_observation_text(
                        record['observation_text'],
                        args.max_observation_entities,
                    )
                if args.dedupe:
                    dedupe_key = make_dedupe_key(record)
                    if dedupe_key in seen:
                        stats['duplicate_rows'] += 1
                        continue
                    seen.add(dedupe_key)
                compact = build_compact_record(
                    record=record,
                    kg=kg,
                    graph_split=split,
                    result_top_k=args.result_top_k,
                    trace_mode=args.trace_mode,
                    trace_cache=trace_cache,
                    dsl_cache=dsl_cache,
                )
                output_file.write(json.dumps(compact, ensure_ascii=False) + '\n')
                stats['written_rows'] += 1
            except Exception as exc:
                stats['error_rows'] += 1
                error_file.write(json.dumps({
                    'line_number': stats['read_rows'],
                    'error': str(exc),
                    'raw': line[:2000],
                }, ensure_ascii=False) + '\n')
                if args.strict:
                    raise
            if stats['read_rows'] % args.log_every == 0:
                progress.set_postfix({
                    'written': stats['written_rows'],
                    'dup': stats['duplicate_rows'],
                    'err': stats['error_rows'],
                })

    os.replace(tmp_output_path, output_path)
    if stats['error_rows'] > 0:
        os.replace(tmp_error_path, error_path)
    elif os.path.exists(tmp_error_path):
        os.remove(tmp_error_path)
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--input-root', default='./sampled_data_surface/')
    parser.add_argument('--output-root', default='./sampled_data_abduction/')
    parser.add_argument('--splits', default='train,valid,test')
    parser.add_argument('--result-top-k', type=int, default=3)
    parser.add_argument('--max-observation-entities', type=int, default=8)
    parser.add_argument('--trace-mode', choices=['eager', 'lazy'], default='lazy')
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--trace-cache-size', type=int, default=20000)
    parser.add_argument('--dsl-cache-size', type=int, default=50000)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--dedupe', dest='dedupe', action='store_true')
    parser.add_argument('--no-dedupe', dest='dedupe', action='store_false')
    parser.set_defaults(dedupe=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.splits = [split.strip() for split in args.splits.split(',') if split.strip()]
    invalid_splits = [split for split in args.splits if split not in {'train', 'valid', 'test'}]
    if invalid_splits:
        raise ValueError(f'Unsupported splits: {invalid_splits}')

    kg = load_kg(args.dataname)
    write_stats(args.output_root, args.dataname, kg)
    split_stats = {}
    for split in args.splits:
        split_stats[split] = convert_split(args, kg, split)
        print(f'# Converted {split}: {split_stats[split]}')
    write_manifest(args, split_stats)


if __name__ == '__main__':
    main()
