import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load import load_kg, resolve_sampled_dataset_path, resolve_stats_path
from utils.textualization import attach_textual_fields


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--input-root', default='./sampled_data/')
    parser.add_argument('--output-root', default='./sampled_data_compact/')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include-graph-field', action='store_true')
    parser.add_argument('--progress-every', type=int, default=100000)
    return parser.parse_args()


def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f'Output directory already exists: {path}')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(output_dir: Path, args, split_stats):
    manifest = {
        'format_version': 'compact_text_v1',
        'dataname': args.dataname,
        'input_root': args.input_root,
        'output_root': args.output_root,
        'include_graph_field': args.include_graph_field,
        'changes': [
            'Removed entity prefix "ent:" from textual fields.',
            'Removed relation prefix "rel:" from textual fields.',
            'Kept relation direction markers "+" and "-" to preserve edge direction.',
            'Kept OBS/COND and control labels for disambiguating input structure.',
        ],
        'split_stats': split_stats,
    }
    with open(output_dir / 'text_format_manifest.json', 'w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)


def process_split(input_path: Path, output_path: Path, kg, include_graph_field: bool, progress_every: int):
    row_count = 0
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for row_count, record in enumerate(iter_jsonl(input_path), start=1):
            processed = attach_textual_fields(
                record,
                kg,
                include_graph_text=include_graph_field,
            )
            output_file.write(json.dumps(processed, ensure_ascii=False) + '\n')
            if progress_every > 0 and row_count % progress_every == 0:
                print(f'  processed {row_count} rows from {input_path.name}')
    print(f'  finished {input_path.name}: {row_count} rows')
    return row_count


def main():
    args = parse_args()

    input_dataset_dir = Path(args.input_root) / args.dataname
    output_dataset_dir = Path(args.output_root) / args.dataname
    ensure_output_dir(output_dataset_dir, overwrite=args.overwrite)

    kg = load_kg(args.dataname)

    split_stats = {}
    for split in args.splits:
        input_path = Path(resolve_sampled_dataset_path(args.input_root, args.dataname, split))
        output_path = output_dataset_dir / input_path.name
        print(f'Processing split={split}')
        split_stats[split] = process_split(
            input_path=input_path,
            output_path=output_path,
            kg=kg,
            include_graph_field=args.include_graph_field,
            progress_every=args.progress_every,
        )

    stats_path = Path(resolve_stats_path(args.input_root, args.dataname))
    shutil.copy2(stats_path, output_dataset_dir / 'stats.txt')

    state_path = input_dataset_dir / 'sampling_state.json'
    if state_path.exists():
        shutil.copy2(state_path, output_dataset_dir / state_path.name)

    write_manifest(output_dataset_dir, args, split_stats)
    print(f'Wrote compact dataset to {output_dataset_dir}')


if __name__ == '__main__':
    main()
