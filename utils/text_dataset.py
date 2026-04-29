from utils.action_supervision import (
    extract_observation_entity_tokens,
    infer_branch_hops,
    infer_action_steps,
    infer_action_type,
    render_action_text,
)
from utils.kg_actions import execute_action_text
from utils.logic_dsl import surface_query_to_dsl
from utils.textualization import attach_textual_fields, tokenize_surface_text


def extract_candidate_targets(result_text: str, top_k: int = 10) -> list[str]:
    targets = []
    for line in str(result_text).splitlines():
        tokens = tokenize_surface_text(line)
        if len(tokens) >= 2 and tokens[0] == 'CANDIDATE':
            targets.append(tokens[1])
        if len(targets) >= top_k:
            break
    return targets


def build_stage2_trace(pattern_str, observation_text, kg, graph_split, top_k):
    action_type = infer_action_type(pattern_str)
    observation_targets = extract_observation_entity_tokens(observation_text)
    targets = observation_targets[:]
    branch_hops = infer_branch_hops(pattern_str)
    steps = max(branch_hops, default=infer_action_steps(pattern_str))
    trace = []
    all_candidates = []

    if targets:
        action = render_action_text(
            action_type=action_type,
            targets=targets,
            top_k=top_k,
        )
        result = execute_action_text(action, kg=kg, graph_split=graph_split)
        trace.append({
            'action': action,
            'result': result,
        })
        targets = extract_candidate_targets(result, top_k=top_k)
        all_candidates.extend(targets)

    for _ in range(2, steps + 1):
        if not targets:
            break
        action = render_action_text(
            action_type='EXPAND',
            targets=targets,
            direction='backward',
            top_k=top_k,
        )
        result = execute_action_text(action, kg=kg, graph_split=graph_split)
        trace.append({
            'action': action,
            'result': result,
        })
        targets = extract_candidate_targets(result, top_k=top_k)
        all_candidates.extend(targets)

    coverage_candidates = []
    seen = set()
    for candidate in all_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        coverage_candidates.append(candidate)
        if len(coverage_candidates) >= top_k:
            break

    if coverage_candidates and observation_targets:
        action = render_action_text(
            action_type='CHECK_COVERAGE',
            candidates=coverage_candidates,
            obs=observation_targets,
            top_k=top_k,
        )
        result = execute_action_text(action, kg=kg, graph_split=graph_split)
        trace.append({
            'action': action,
            'result': result,
        })

    return trace


def build_minimal_text_record(
        record,
        kg,
        sample_id,
        graph_split: str | None = None,
        result_top_k: int = 10):
    del sample_id
    enriched = attach_textual_fields(record, kg)
    observation_text = enriched['observation_text']
    logic_dsl = surface_query_to_dsl(enriched['hypothesis_text'], kg)
    stage2_trace = []
    if graph_split is not None:
        stage2_trace = build_stage2_trace(
            pattern_str=record['pattern_str'],
            observation_text=observation_text,
            kg=kg,
            graph_split=graph_split,
            top_k=result_top_k,
        )

    return {
        'pattern_str': record['pattern_str'],
        'observation_text': observation_text,
        'logic_dsl': logic_dsl,
        'stage2_trace': stage2_trace,
    }
