import argparse
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from model.tokenizer import create_text_tokenizer, get_text_extra_tokens
from model.transformer import get_tokenizer_path, resolve_model_runtime_config
from scripts.run_stage2_tool_loop import load_stage2_model, run_loop
from utils.load import load_kg, load_yaml, resolve_sampled_dataset_path
from utils.rl_rewards import score_rollout_trajectory


def iter_rows(path, max_rows=0):
    with open(path, 'r', encoding='utf-8') as input_file:
        for row_index, line in enumerate(input_file, start=1):
            if max_rows > 0 and row_index > max_rows:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def evaluate_rollouts(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    kg = load_kg(args.dataname)
    config_model = load_yaml(args.config_model)
    model_runtime_config = resolve_model_runtime_config(args.modelname, config_model)
    tokenizer, ntoken = create_text_tokenizer(
        get_tokenizer_path(model_runtime_config),
        extra_tokens=get_text_extra_tokens(include_graph_tokens=False),
        closed_text_tokens=None,
        trust_remote_code=bool(model_runtime_config.get('trust_remote_code', False)),
    )
    model = load_stage2_model(args, tokenizer, ntoken, device)
    data_path = resolve_sampled_dataset_path(args.data_root, args.dataname, args.split)

    rows = []
    for record in iter_rows(data_path, max_rows=args.max_rows):
        # Keep run_loop input explicit; args.observation is a CLI field, not per-row state.
        row_args = argparse.Namespace(**vars(args))
        row_args.observation = record['observation_text']
        rollout = run_loop(
            args=row_args,
            model=model,
            tokenizer=tokenizer,
            kg=kg,
            device=device,
        )
        score = score_rollout_trajectory(
            rollout=rollout,
            target=record['logic_dsl'],
            observation_text=record['observation_text'],
            kg=kg,
            graph_samplers=kg.graph_samplers,
            graph_split=args.graph_split,
        )
        rows.append({
            'observation_text': record['observation_text'],
            'gold_dsl': record['logic_dsl'],
            'pred_dsl': rollout.get('dsl', ''),
            'stopped_by': rollout.get('stopped_by', ''),
            'num_history_lines': len(rollout.get('history', [])),
            **score,
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    if not rows:
        print('# No rows evaluated.')
        return
    metric_names = [
        'stage3_reward',
        'logic_stage3_reward',
        'validity',
        'jaccard',
        'dice',
        'overlap',
        'smatch',
        'answer_precision',
        'answer_recall',
        'action_reward_avg',
        'action_parse_rate',
        'action_execution_rate',
        'num_actions',
    ]
    summary = {
        metric: sum(float(row.get(metric, 0.0)) for row in rows) / len(rows)
        for metric in metric_names
    }
    summary['num_rows'] = len(rows)
    print('# SUMMARY')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./sampled_data_abduction/')
    parser.add_argument('--split', default='train')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--modelname', default='GPT2_6_act_nt')
    parser.add_argument('--config-model', default='configs/config-model.yml')
    parser.add_argument('--config-train', default='configs/config-train.yml')
    parser.add_argument('--checkpoint-root', default='./ckpt/')
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--scale', default='default')
    parser.add_argument('--max-answer-size', type=int, default=8)
    parser.add_argument('--graph-split', default='train')
    parser.add_argument('--max-action-steps', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--use-pretrained-text-model', action='store_true')
    parser.add_argument('--device', default='')
    parser.add_argument('--print-raw', action='store_true')
    parser.add_argument('--max-rows', type=int, default=10)
    return parser.parse_args()


def main():
    evaluate_rollouts(parse_args())


if __name__ == '__main__':
    main()
