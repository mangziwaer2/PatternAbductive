import argparse
import json
import random
from pathlib import Path

from utils.action_supervision import (
    extract_observation_entity_tokens,
    infer_action_steps,
    infer_action_type,
    infer_branch_hops,
    render_action_text,
)
from utils.evidence import render_evidence_package
from utils.kg_actions import execute_action, parse_action_text
from utils.load import load_kg
from utils.logic_dsl import pattern_str_to_pattern_dsl, tag_dsl_text
from utils.text_dataset import extract_candidate_targets, result_has_edges
from utils.textualization import format_surface_atom, normalize_symbol_name


def read_jsonl_record(path: str, row_index: int) -> dict:
    with open(path, 'r', encoding='utf-8') as input_file:
        for index, line in enumerate(input_file):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f'Row index {row_index} out of range: {path}')


def normalize_observation_text(obs: str) -> str:
    obs = str(obs or '').strip()
    if not obs:
        raise ValueError('OBS is required.')
    if obs.startswith('OBS '):
        return obs
    return f'OBS {obs}'


def entity_text(kg, entity_id: int) -> str:
    return format_surface_atom(normalize_symbol_name(kg.ent_id2name[int(entity_id)]))


def relation_text(kg, relation_id: int) -> str:
    return format_surface_atom(normalize_symbol_name(kg.rel_id2name[int(relation_id)]))


def edge_text(kg, edge: dict) -> str:
    return (
        f'{entity_text(kg, edge["subject_id"])} '
        f'--{relation_text(kg, edge["relation_id"])}--> '
        f'{entity_text(kg, edge["object_id"])}'
    )


def print_evidence_debug(evidence: dict, kg, dump_json: bool = False):
    print(f'mode: {evidence.get("mode", "")}')
    print(f'candidate_count: {len(evidence.get("candidates", []))}')
    observation_ids = evidence.get('observation_ids') or []
    if observation_ids:
        print('observation_ids:', ', '.join(
            f'{entity_text(kg, entity_id)}#{entity_id}' for entity_id in observation_ids
        ))

    for index, candidate in enumerate(evidence.get('candidates', []), start=1):
        candidate_id = candidate.get('entity_id')
        if candidate_id is not None:
            print(f'candidate {index}: {entity_text(kg, candidate_id)}#{candidate_id}')
        supports = candidate.get('support', [])
        for support in supports:
            print(f'  support: {edge_text(kg, support)}')

    if dump_json:
        print('# raw evidence json')
        print(json.dumps(evidence, ensure_ascii=False, indent=2))


def build_debug_trace(
        pattern_str: str,
        observation_text: str,
        kg,
        graph_split: str,
        top_k: int,
        max_steps_override: int = 0,
        dump_evidence_json: bool = False):
    action_type = infer_action_type(pattern_str)
    branch_hops = infer_branch_hops(pattern_str)
    inferred_steps = max(branch_hops, default=infer_action_steps(pattern_str))
    steps = max_steps_override if max_steps_override > 0 else inferred_steps
    observation_targets = extract_observation_entity_tokens(observation_text)
    targets = observation_targets[:]
    trace = []

    print('# Input')
    print(f'pattern_str: {pattern_str}')
    print(f'pattern_dsl: {pattern_str_to_pattern_dsl(pattern_str)}')
    print(f'observation: {observation_text}')
    print(f'observation_targets: {observation_targets}')
    print(f'inferred_first_action: {action_type}')
    print(f'branch_hops: {branch_hops}')
    print(f'inferred_steps: {inferred_steps}')
    print(f'used_steps: {steps}')
    print(f'top_k/sample_k: {top_k}')

    for step in range(1, steps + 1):
        if not targets:
            print(f'\n# Step {step}: stop, no targets for next action.')
            break

        if step == 1:
            current_action_type = action_type
            action = render_action_text(
                action_type=current_action_type,
                targets=targets,
                top_k=top_k,
            )
            reason = (
                'first action is inferred from pattern: '
                'n -> FIND_EXCLUSION, u -> FIND_ALTERNATIVE, otherwise FIND_COMMON'
            )
        else:
            current_action_type = 'EXPAND'
            action = render_action_text(
                action_type=current_action_type,
                targets=targets,
                direction='backward',
                top_k=top_k,
            )
            reason = 'follow-up action expands candidates extracted from the previous RESULT'

        print(f'\n# Step {step}: {current_action_type}')
        print(f'reason: {reason}')
        print(f'targets: {targets}')
        print(action)

        action_dict = parse_action_text(action)
        evidence = execute_action(action_dict, kg=kg, graph_split=graph_split)
        print('\n# Evidence construction')
        print_evidence_debug(evidence, kg=kg, dump_json=dump_evidence_json)

        result = render_evidence_package(evidence, kg, prefix='RESULT')
        print('\n# RESULT')
        print(result)

        if not result_has_edges(result):
            print(f'# Step {step}: stop, RESULT has no edges.')
            break

        trace.append({
            'action': action,
            'result': result,
        })
        targets = extract_candidate_targets(result, top_k=top_k)
        print(f'# Next targets extracted from RESULT subject nodes: {targets}')

    return trace


def print_sft_view(observation_text: str, trace: list[dict], logic_dsl: str | None):
    print('\n# Stage2 Prefix-to-Next-Step SFT View')
    history_parts = []
    for index, event in enumerate(trace, start=1):
        source = '\n'.join([observation_text, *history_parts])
        print(f'\n## sample {index}: predict ACTION')
        print('SOURCE:')
        print(source)
        print('TARGET:')
        print(event['action'])
        history_parts.extend([event['action'], event['result']])

    print('\n## final sample: predict DSL')
    print('SOURCE:')
    print('\n'.join([observation_text, *history_parts]))
    print('TARGET:')
    if logic_dsl:
        print(tag_dsl_text(logic_dsl))
    else:
        print('# Concrete DSL is not derivable from pattern + OBS alone.')
        print('# Pass --dsl or --row-path to show the real final DSL target.')


def main():
    parser = argparse.ArgumentParser(description='Debug pattern+OBS -> ACTION/RESULT trace construction.')
    parser.add_argument('--pattern', default='(i,(n,(p,(e))),(p,(e)))')
    parser.add_argument('--obs', default='OBS [David Mainz]')
    parser.add_argument('--dsl', default='')
    parser.add_argument('--row-path', default='', help='Optional sampled jsonl path containing pattern_str/observation_text/logic_dsl.')
    parser.add_argument('--row-index', type=int, default=153)
    parser.add_argument('--dataname', default='DBpedia50')
    parser.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dump-evidence-json', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.row_path:
        record = read_jsonl_record(args.row_path, args.row_index)
        pattern_str = record.get('pattern_str') or args.pattern
        observation_text = record.get('observation_text') or args.obs
        logic_dsl = record.get('logic_dsl') or args.dsl
        print(f'# Loaded row {args.row_index} from {Path(args.row_path).resolve()}')
    else:
        pattern_str = args.pattern
        observation_text = args.obs
        logic_dsl = args.dsl

    if not pattern_str:
        raise ValueError('Missing --pattern or --row-path.')
    observation_text = normalize_observation_text(observation_text)
    logic_dsl = str(logic_dsl or '').strip()

    kg = load_kg(args.dataname)
    trace = build_debug_trace(
        pattern_str=pattern_str,
        observation_text=observation_text,
        kg=kg,
        graph_split=args.split,
        top_k=args.top_k,
        max_steps_override=args.max_steps,
        dump_evidence_json=args.dump_evidence_json,
    )
    print_sft_view(
        observation_text=observation_text,
        trace=trace,
        logic_dsl=logic_dsl,
    )


if __name__ == '__main__':
    main()
