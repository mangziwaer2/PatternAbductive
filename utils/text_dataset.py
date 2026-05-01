from utils.action_supervision import (
    extract_observation_entity_tokens,
    ACTION_SCHEMA_VERSION,
    infer_branch_hops,
    infer_action_steps,
    infer_action_type,
    render_action_text,
)
from utils.kg_actions import execute_action_text
from utils.logic_dsl import surface_query_to_dsl
from utils.evidence import RESULT_END_TAG, RESULT_START_TAG, strip_result_tags
from utils.textualization import attach_textual_fields, tokenize_surface_text


def extract_candidate_targets(result_text: str, top_k: int = 10) -> list[str]:
    targets = []
    for line in strip_result_tags(result_text).splitlines():
        if line in {RESULT_START_TAG, RESULT_END_TAG, 'RESULT'}:
            continue
        tokens = tokenize_surface_text(line)
        if len(tokens) >= 5 and tokens[1] == '--' and tokens[3] == '-->':
            targets.append(tokens[0])
        if len(targets) >= top_k:
            break
    deduped = []
    seen = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        deduped.append(target)
    return deduped


def result_has_edges(result_text: str) -> bool:
    for line in strip_result_tags(result_text).splitlines():
        if line in {RESULT_START_TAG, RESULT_END_TAG, 'RESULT'}:
            continue
        tokens = tokenize_surface_text(line)
        if len(tokens) >= 5 and tokens[1] == '--' and tokens[3] == '-->':
            return True
    return False


def build_stage2_trace(pattern_str, observation_text, kg, graph_split, top_k):
    action_type = infer_action_type(pattern_str)
    observation_targets = extract_observation_entity_tokens(observation_text)
    targets = observation_targets[:]
    branch_hops = infer_branch_hops(pattern_str)
    steps = max(branch_hops, default=infer_action_steps(pattern_str))
    trace = []

    if targets:
        action = render_action_text(
            action_type=action_type,
            targets=targets,
            top_k=top_k,
        )
        result = execute_action_text(action, kg=kg, graph_split=graph_split)
        if not result_has_edges(result):
            return []
        trace.append({
            'schema': ACTION_SCHEMA_VERSION,
            'action': action,
            'result': result,
        })
        targets = extract_candidate_targets(result, top_k=top_k)

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
        if not result_has_edges(result):
            break
        trace.append({
            'schema': ACTION_SCHEMA_VERSION,
            'action': action,
            'result': result,
        })
        targets = extract_candidate_targets(result, top_k=top_k)

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
        if not stage2_trace:
            return None

    return {
        'pattern_str': record['pattern_str'],
        'observation_text': observation_text,
        'logic_dsl': logic_dsl,
        'stage2_trace': stage2_trace,
    }
