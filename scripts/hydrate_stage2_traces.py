import argparse
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
from utils.text_dataset import build_stage2_trace


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


def copy_sidecar_files(input_root, output_root, dataname):
    input_dir = os.path.join(input_root, dataname)
    output_dir = os.path.join(output_root, dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name in ['stats.txt', 'text_format_manifest.json']:
        src = os.path.join(input_dir, name)
        dst = os.path.join(output_dir, name)
        if not os.path.exists(src):
            continue
        with open(src, 'r', encoding='utf-8') as input_file:
            content = input_file.read()
        with open(dst, 'w', encoding='utf-8') as output_file:
            output_file.write(content)


def hydrate_split(args, kg, split):
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

    trace_cache = BoundedCache(args.trace_cache_size)
    stats = {
        'read_rows': 0,
        'written_rows': 0,
        'hydrated_rows': 0,
        'kept_rows': 0,
        'error_rows': 0,
        'input_path': input_path,
        'output_path': output_path,
    }

    with open(input_path, 'r', encoding='utf-8') as input_file, \
            open(tmp_output_path, 'w', encoding='utf-8') as output_file, \
            open(tmp_error_path, 'w', encoding='utf-8') as error_file:
        progress = tqdm(input_file, desc=f'hydrate_{split}', unit='row')
        for line in progress:
            if args.max_rows > 0 and stats['read_rows'] >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            stats['read_rows'] += 1
            try:
                record = json.loads(line)
                trace = record.get('stage2_trace') or []
                if trace and not args.rebuild:
                    stats['kept_rows'] += 1
                else:
                    key = (
                        split,
                        record['pattern_str'],
                        record['observation_text'],
                        int(args.result_top_k),
                    )
                    cached = trace_cache.get(key)
                    if cached is None:
                        cached = build_stage2_trace(
                            pattern_str=record['pattern_str'],
                            observation_text=record['observation_text'],
                            kg=kg,
                            graph_split=split,
                            top_k=args.result_top_k,
                        )
                        trace_cache.put(key, cached)
                    record['stage2_trace'] = cached
                    stats['hydrated_rows'] += 1
                output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
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
                    'hydrated': stats['hydrated_rows'],
                    'err': stats['error_rows'],
                })

    os.replace(tmp_output_path, output_path)
    if stats['error_rows'] > 0:
        os.replace(tmp_error_path, error_path)
    elif os.path.exists(tmp_error_path):
        os.remove(tmp_error_path)
    return stats


def write_manifest(args, split_stats):
    output_dir = os.path.join(args.output_root, args.dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'trace_hydration_manifest.json')
    manifest = {
        'generated_by': 'scripts/hydrate_stage2_traces.py',
        'input_root': args.input_root,
        'output_root': args.output_root,
        'dataname': args.dataname,
        'splits': args.splits,
        'result_top_k': args.result_top_k,
        'max_rows': args.max_rows,
        'rebuild': args.rebuild,
        'split_stats': split_stats,
        'notes': [
            'stage2_trace is precomputed to avoid KG calls during Stage 2 SFT preprocessing.',
            'SFT still expands trace into prefix-to-next-step samples inside utils/dataloader.py.',
            'RL rollout training does not use stage2_trace as target.',
        ],
    }
    with open(manifest_path, 'w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--input-root', default='./sampled_data_abduction/')
    parser.add_argument('--output-root', default='./sampled_data_abduction_traced/')
    parser.add_argument('--splits', default='train,valid,test')
    parser.add_argument('--result-top-k', type=int, default=3)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--trace-cache-size', type=int, default=50000)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    args.splits = [split.strip() for split in args.splits.split(',') if split.strip()]
    invalid_splits = [split for split in args.splits if split not in {'train', 'valid', 'test'}]
    if invalid_splits:
        raise ValueError(f'Unsupported splits: {invalid_splits}')

    kg = load_kg(args.dataname)
    copy_sidecar_files(args.input_root, args.output_root, args.dataname)
    split_stats = {}
    for split in args.splits:
        split_stats[split] = hydrate_split(args, kg, split)
        print(json.dumps({split: split_stats[split]}, ensure_ascii=False, indent=2))
    write_manifest(args, split_stats)


if __name__ == '__main__':
    main()
