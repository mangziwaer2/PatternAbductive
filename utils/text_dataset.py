from utils.action_supervision import (
    extract_observation_entity_tokens,
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
    targets = extract_observation_entity_tokens(observation_text)
    steps = infer_action_steps(pattern_str)
    trace = []

    for _ in range(steps):
        if not targets:
            break
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
