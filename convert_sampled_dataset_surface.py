import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from utils.condition import DEFAULT_EXCLUDED_CONDITION_TYPES, normalize_condition_type_list
from utils.load import load_kg
from utils.text_dataset import build_minimal_text_record


def should_keep_record(record, excluded_condition_types):
    signature = str(record.get('condition_signature', 'unconditional') or 'unconditional')
    if signature == 'unconditional':
        return True
    present_types = {token for token in signature.split('+') if token}
    return len(present_types.intersection(excluded_condition_types)) == 0


def resolve_split_path(root, dataname, split):
    return os.path.join(root, dataname, f'{dataname}-{split}-a2q.jsonl')


def count_lines(path):
    with open(path, 'r', encoding='utf-8') as input_file:
        return sum(1 for _ in input_file)


def convert_split(args, kg, split):
    input_path = resolve_split_path(args.input_root, args.dataname, split)
    output_dir = os.path.join(args.output_root, args.dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.dataname}-{split}-a2q.jsonl')

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f'Output already exists: {output_path}. Use --overwrite to replace it.')

    kept_rows = 0
    total_lines = args.max_rows if args.max_rows > 0 else count_lines(input_path)

    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
        with tqdm(total=total_lines, desc=f'convert_{split}') as progress:
            for row_index, line in enumerate(input_file, start=1):
                if args.max_rows > 0 and row_index > args.max_rows:
                    break
                progress.update(1)
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not should_keep_record(record, args.exclude_condition_types):
                    continue
                minimal_record = build_minimal_text_record(
                    record=record,
                    kg=kg,
                    sample_id=kept_rows,
                    graph_split=split if args.include_kg_hints else None,
                    include_kg_hints=args.include_kg_hints,
                    kg_hints_max_facts=args.kg_hints_max_facts,
                )
                output_file.write(json.dumps(minimal_record, ensure_ascii=False) + '\n')
                kept_rows += 1

    return output_path, kept_rows


def write_stats(output_dir, kg):
    stats_path = os.path.join(output_dir, 'stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        stats_file.write(f'num_ent\t{kg.num_ent}\n')
        stats_file.write(f'num_rel\t{kg.num_rel}\n')


def write_manifest(args, output_dir):
    manifest_path = os.path.join(output_dir, 'text_format_manifest.json')
    manifest = {
        'format_version': 'surface_text_minimal_v1',
        'generated_by': 'convert_sampled_dataset_surface.py',
        'input_root': args.input_root,
        'output_root': args.output_root,
        'dataname': args.dataname,
        'include_kg_hints': args.include_kg_hints,
        'kg_hints_max_facts': args.kg_hints_max_facts,
        'exclude_condition_types': sorted(args.exclude_condition_types),
        'changes': [
            'Kept only fields required by current text-to-text training.',
            'Entity and relation names use surface forms with spaces instead of underscores.',
            'Logical hypotheses use surface operators like (-p), (-e), (-i), (-u), (-n).',
            'Entities and relations are wrapped with square brackets for parse-safe text generation.',
            'Relations keep their +/- direction sign inside brackets, e.g. [+team] and [-team].',
        ],
    }
    with open(manifest_path, 'w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--input-root', default='./sampled_data_compact/')
    parser.add_argument('--output-root', default='./sampled_data_surface/')
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--include-kg-hints', action='store_true')
    parser.add_argument('--kg-hints-max-facts', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--exclude-condition-types',
        default=','.join(sorted(DEFAULT_EXCLUDED_CONDITION_TYPES)),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.exclude_condition_types = set(normalize_condition_type_list(args.exclude_condition_types))

    kg = load_kg(args.dataname)
    output_dir = os.path.join(args.output_root, args.dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_stats(output_dir, kg)
    write_manifest(args, output_dir)

    for split in ['train', 'valid', 'test']:
        output_path, kept_rows = convert_split(args, kg, split)
        print(f'# Converted split={split}: kept_rows={kept_rows}, output={output_path}')


if __name__ == '__main__':
    main()
