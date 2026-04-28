import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.execution import execute_logic_text
from utils.load import load_kg, resolve_sampled_dataset_path
from utils.textualization import observation_text_to_answer_ids


def resolve_output_path(output_root, dataname, split):
    output_dir = os.path.join(output_root, dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(output_dir, f'{dataname}-{split}-a2q.jsonl')


def copy_sidecar_files(input_root, output_root, dataname):
    input_dir = os.path.join(input_root, dataname)
    output_dir = os.path.join(output_root, dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name in [
        'stats.txt',
        'text_format_manifest.json',
        'trace_hydration_manifest.json',
    ]:
        source_path = os.path.join(input_dir, name)
        target_path = os.path.join(output_dir, name)
        if os.path.exists(source_path):
            shutil.copyfile(source_path, target_path)


def init_split_stats(split, input_path, output_path=None):
    return {
        'split': split,
        'input_path': input_path,
        'output_path': output_path or '',
        'read_rows': 0,
        'kept_rows': 0,
        'failed_rows': 0,
        'missing_field_rows': 0,
        'parse_error_rows': 0,
        'execution_error_rows': 0,
        'empty_observation_rows': 0,
        'full_cover_rows': 0,
        'partial_cover_rows': 0,
        'zero_cover_rows': 0,
        'total_obs_count': 0,
        'total_obs_hit_count': 0,
        'total_answer_count': 0,
        'average_obs_recall': 0.0,
        'average_answer_count': 0.0,
    }


def _safe_names(entity_ids, kg, limit=16):
    names = []
    for entity_id in list(entity_ids)[:limit]:
        names.append(kg.ent_id2name.get(int(entity_id), str(entity_id)))
    return names


def _score_row(record, kg, execution_split):
    if 'observation_text' not in record or 'logic_dsl' not in record:
        return {
            'status': 'missing_field',
            'obs_ids': [],
            'answers': [],
            'obs_hit_count': 0,
            'obs_count': 0,
            'obs_recall': 0.0,
            'answer_count': 0,
            'error': 'missing observation_text or logic_dsl',
        }

    try:
        obs_ids = observation_text_to_answer_ids(record['observation_text'], kg)
    except Exception as exc:
        return {
            'status': 'parse_error',
            'obs_ids': [],
            'answers': [],
            'obs_hit_count': 0,
            'obs_count': 0,
            'obs_recall': 0.0,
            'answer_count': 0,
            'error': f'observation_parse_error: {exc}',
        }

    execution = execute_logic_text(
        logic_text=record['logic_dsl'],
        kg=kg,
        graph_samplers=kg.graph_samplers,
        split=execution_split,
    )
    answers = execution.get('answers') or []
    obs_set = set(obs_ids)
    answer_set = set(answers)
    hit_count = len(obs_set & answer_set)
    obs_count = len(obs_set)
    obs_recall = hit_count / max(obs_count, 1)

    if execution.get('parse_success', 0.0) != 1.0:
        status = 'parse_error'
    elif execution.get('execution_success', 0.0) != 1.0:
        status = 'execution_error'
    elif obs_count == 0:
        status = 'empty_observation'
    elif hit_count == 0:
        status = 'zero_cover'
    elif hit_count < obs_count:
        status = 'partial_cover'
    else:
        status = 'full_cover'

    return {
        'status': status,
        'obs_ids': obs_ids,
        'answers': answers,
        'obs_hit_count': hit_count,
        'obs_count': obs_count,
        'obs_recall': obs_recall,
        'answer_count': len(answer_set),
        'error': execution.get('error', ''),
    }


def _diagnose_splits(record, kg, splits):
    diagnostics = {}
    for split in splits:
        diagnostics[split] = {
            key: value
            for key, value in _score_row(record, kg, split).items()
            if key in {'status', 'obs_hit_count', 'obs_count', 'obs_recall', 'answer_count', 'error'}
        }
    return diagnostics


def _write_failure(failure_file, row_index, record, score, kg, diagnose=None):
    failure = {
        'row_index': row_index,
        'status': score['status'],
        'obs_recall': score['obs_recall'],
        'obs_hit_count': score['obs_hit_count'],
        'obs_count': score['obs_count'],
        'answer_count': score['answer_count'],
        'error': score.get('error', ''),
        'observation_text': record.get('observation_text', ''),
        'logic_dsl': record.get('logic_dsl', ''),
        'obs_names': _safe_names(score.get('obs_ids', []), kg),
        'hit_names': _safe_names(set(score.get('obs_ids', [])) & set(score.get('answers', [])), kg),
        'answer_names': _safe_names(score.get('answers', []), kg),
    }
    if diagnose:
        failure['diagnose_splits'] = diagnose
    failure_file.write(json.dumps(failure, ensure_ascii=False) + '\n')


def _update_stats(stats, score, kept):
    stats['kept_rows' if kept else 'failed_rows'] += 1
    status_key = f'{score["status"]}_rows'
    if status_key in stats:
        stats[status_key] += 1
    stats['total_obs_count'] += score['obs_count']
    stats['total_obs_hit_count'] += score['obs_hit_count']
    stats['total_answer_count'] += score['answer_count']


def validate_split(args, kg, split):
    input_path = resolve_sampled_dataset_path(args.data_root, args.dataname, split)
    execution_split = split if args.execution_split == 'same' else args.execution_split
    output_path = resolve_output_path(args.output_root, args.dataname, split) if args.output_root else None
    failure_path = None
    if args.failure_root:
        failure_dir = os.path.join(args.failure_root, args.dataname)
        Path(failure_dir).mkdir(parents=True, exist_ok=True)
        failure_path = os.path.join(failure_dir, f'{args.dataname}-{split}-failures.jsonl')

    if output_path and os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f'Output exists: {output_path}. Use --overwrite to replace it.')
    for path in [output_path, failure_path]:
        if path and args.overwrite and os.path.exists(path):
            os.remove(path)

    stats = init_split_stats(split=split, input_path=input_path, output_path=output_path)
    output_file = open(output_path, 'w', encoding='utf-8') if output_path else None
    failure_file = open(failure_path, 'w', encoding='utf-8') if failure_path else None

    try:
        with open(input_path, 'r', encoding='utf-8') as input_file:
            progress = tqdm(input_file, desc=f'validate_{split}', unit='row')
            for row_index, line in enumerate(progress):
                if args.max_rows > 0 and stats['read_rows'] >= args.max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                stats['read_rows'] += 1
                try:
                    record = json.loads(line)
                except Exception as exc:
                    stats['failed_rows'] += 1
                    stats['parse_error_rows'] += 1
                    if failure_file:
                        failure_file.write(json.dumps({
                            'row_index': row_index,
                            'status': 'json_parse_error',
                            'error': str(exc),
                            'raw': line[:2000],
                        }, ensure_ascii=False) + '\n')
                    continue

                score = _score_row(record, kg, execution_split)
                kept = (
                    score['status'] not in {'missing_field', 'parse_error', 'execution_error', 'empty_observation'}
                    and score['obs_recall'] >= args.min_obs_recall
                )
                _update_stats(stats, score, kept)

                if kept and output_file:
                    output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                elif (not kept) and failure_file:
                    diagnose = None
                    if args.diagnose_splits:
                        diagnose = _diagnose_splits(record, kg, args.diagnose_splits)
                    _write_failure(failure_file, row_index, record, score, kg, diagnose=diagnose)

                if stats['read_rows'] % args.log_every == 0:
                    progress.set_postfix({
                        'kept': stats['kept_rows'],
                        'failed': stats['failed_rows'],
                        'full': stats['full_cover_rows'],
                    })
    finally:
        if output_file:
            output_file.close()
        if failure_file:
            failure_file.close()

    if stats['read_rows'] > 0:
        stats['average_obs_recall'] = stats['total_obs_hit_count'] / max(stats['total_obs_count'], 1)
        stats['average_answer_count'] = stats['total_answer_count'] / stats['read_rows']
    return stats


def write_manifest(args, split_stats):
    if not args.output_root:
        return
    output_dir = os.path.join(args.output_root, args.dataname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {
        'generated_by': 'scripts/validate_abduction_dataset.py',
        'input_root': args.data_root,
        'output_root': args.output_root,
        'dataname': args.dataname,
        'splits': args.splits,
        'execution_split': args.execution_split,
        'min_obs_recall': args.min_obs_recall,
        'max_rows': args.max_rows,
        'split_stats': split_stats,
        'notes': [
            'Rows are kept when the gold logic_dsl executes on the selected KG split and covers enough OBS entities.',
            'This does not require ACTION/RESULT traces to contain the full hypothesis path.',
            'Use this filtered root for RL/reward-sensitive experiments when split consistency matters.',
        ],
    }
    with open(os.path.join(output_dir, 'validation_manifest.json'), 'w', encoding='utf-8') as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--data-root', default='./sampled_data_abduction_traced/')
    parser.add_argument('--output-root', default='')
    parser.add_argument('--failure-root', default='./results/dataset_validation/')
    parser.add_argument('--splits', default='train,valid,test')
    parser.add_argument('--execution-split', choices=['same', 'train', 'valid', 'test'], default='same')
    parser.add_argument('--min-obs-recall', type=float, default=1.0)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=10000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--diagnose-splits',
        default='',
        help='Comma-separated split list for failure diagnostics, e.g. train,valid,test.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.splits = [split.strip() for split in args.splits.split(',') if split.strip()]
    args.diagnose_splits = [split.strip() for split in args.diagnose_splits.split(',') if split.strip()]
    valid_splits = {'train', 'valid', 'test'}
    invalid_splits = [split for split in [*args.splits, *args.diagnose_splits] if split not in valid_splits]
    if invalid_splits:
        raise ValueError(f'Unsupported splits: {invalid_splits}')

    kg = load_kg(args.dataname)
    if args.output_root:
        copy_sidecar_files(args.data_root, args.output_root, args.dataname)

    split_stats = {}
    for split in args.splits:
        split_stats[split] = validate_split(args, kg, split)
        print(json.dumps({split: split_stats[split]}, ensure_ascii=False, indent=2))
    write_manifest(args, split_stats)


if __name__ == '__main__':
    main()
