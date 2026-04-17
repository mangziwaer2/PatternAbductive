import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader import REPRESENTATION_TEXT, new_create_dataset
from utils.load import (
    resolve_processed_dataset_cache_path,
    resolve_sampled_dataset_path,
    save_processed_dataset_to_disk,
)
from utils.load import load_kg


def parse_args():
    parser = argparse.ArgumentParser(description='Build reusable processed HuggingFace dataset cache.')
    parser.add_argument('--data_root', default='./sampled_data_compact/')
    parser.add_argument('--dataname', default='DBpedia50')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--pattern_path', default='./metadata/pattern_filtered.csv')
    parser.add_argument('--representation', default='text')
    parser.add_argument('--source_text_field', default='observation_text')
    parser.add_argument('--target_text_field', default='hypothesis_text')
    parser.add_argument('--dataset_cache_root', default='./dataset_cache/')
    parser.add_argument('--dataset_num_proc', type=int, default=1)
    parser.add_argument('--dataset_map_batch_size', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def iter_jsonl(path: Path):
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dataset_needs_kg(args):
    if args.representation != REPRESENTATION_TEXT:
        return True

    required_fields = {args.source_text_field, args.target_text_field}
    for split in args.splits:
        sample_path = Path(resolve_sampled_dataset_path(args.data_root, args.dataname, split))
        first_record = next(iter_jsonl(sample_path), None)
        if first_record is None:
            continue
        if not required_fields.issubset(first_record.keys()):
            return True
    return False


def main():
    args = parse_args()
    pattern_filtered = pd.read_csv(args.pattern_path, index_col='id')

    kg = None
    if dataset_needs_kg(args):
        print('# Loading KG for textual fallback preprocessing')
        kg = load_kg(args.dataname)

    dataset_dict, _, _ = new_create_dataset(
        dataname=args.dataname,
        pattern_filtered=pattern_filtered,
        data_root=args.data_root,
        splits=args.splits,
        max_rows_by_split={},
        kg=kg,
        source_text_field=args.source_text_field,
        target_text_field=args.target_text_field,
        representation=args.representation,
        dataset_cache_root=args.dataset_cache_root,
        dataset_num_proc=args.dataset_num_proc,
        dataset_map_batch_size=args.dataset_map_batch_size,
        prefer_saved_processed_cache=not args.overwrite,
    )

    print('# Saving processed dataset cache')
    for split in args.splits:
        output_path = save_processed_dataset_to_disk(
            dataset=dataset_dict[split],
            dataname=args.dataname,
            split=split,
            representation=args.representation,
            source_text_field=args.source_text_field,
            target_text_field=args.target_text_field,
            dataset_cache_root=args.dataset_cache_root,
            overwrite=args.overwrite,
        )
        print(f'  split={split} rows={len(dataset_dict[split])} path={output_path}')

    print('# Processed cache ready')
    for split in args.splits:
        output_path = resolve_processed_dataset_cache_path(
            dataname=args.dataname,
            split=split,
            representation=args.representation,
            source_text_field=args.source_text_field,
            target_text_field=args.target_text_field,
            dataset_cache_root=args.dataset_cache_root,
        )
        print(f'  {split}: {output_path}')


if __name__ == '__main__':
    main()
