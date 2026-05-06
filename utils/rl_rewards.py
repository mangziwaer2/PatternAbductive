from utils.action_scoring import score_action_text
from utils.execution import compute_answer_set_scores, execute_logic_text
from utils.textualization import observation_text_to_answer_ids
from utils.tool_loop import extract_action_text, extract_dsl_text
from utils.trace_scoring import (
    score_action_strictness,
    score_bad_generation,
    score_evidence_usage,
    score_step_control,
    score_trace_dsl_consistency,
)


def extract_observation_from_context(source: str) -> str:
    for line in str(source).splitlines():
        line = line.strip()
        if line.startswith('OBS '):
            return line
    return str(source).splitlines()[0].strip() if str(source).splitlines() else str(source).strip()


def score_action_completion(completion: str, target: str, source: str, kg, graph_split: str = 'train') -> dict:
    del target
    action_text = extract_action_text(completion) or str(completion).strip()
    score = score_action_text(
        action_text=action_text,
        observation_text=source,
        kg=kg,
        graph_split=graph_split,
    )
    return {
        'target_type': 'action',
        **score,
        'stage3_reward': float(score['action_reward']),
    }


def score_logic_completion(
        completion: str,
        source: str,
        kg,
        graph_samplers,
        graph_split: str = 'train') -> dict:
    dsl_text = extract_dsl_text(completion) or str(completion).strip()
    observation_text = extract_observation_from_context(source)
    execution = execute_logic_text(
        logic_text=dsl_text,
        kg=kg,
        graph_samplers=graph_samplers,
        split=graph_split,
    )

    try:
        observation_answers = observation_text_to_answer_ids(observation_text, kg)
    except Exception:
        observation_answers = []
    answer_scores = compute_answer_set_scores(execution['answers'], observation_answers)

    compactness = execution['execution_success'] * (
        1.0 / (
            1.0
            + max(execution['answer_count'] - len(observation_answers), 0)
            / max(len(observation_answers), 1)
        )
    )
    complexity_penalty = min(execution['query_complexity'] / 50.0, 1.0)

    reward = (
        0.7 * execution['parse_success']
        + 0.7 * execution['execution_success']
        + 2.0 * answer_scores['answer_jaccard']
        + 0.8 * answer_scores['answer_recall']
        + 0.6 * answer_scores['answer_precision']
        + 0.2 * compactness
        - 0.1 * complexity_penalty
    )
    return {
        'target_type': 'dsl',
        **execution,
        **answer_scores,
        'validity': float(execution['parse_success'] * execution['execution_success']),
        'jaccard': answer_scores['answer_jaccard'],
        'dice': answer_scores['answer_dice'],
        'overlap': answer_scores['answer_overlap'],
        'compactness': float(compactness),
        'complexity_penalty': float(complexity_penalty),
        'stage3_reward': float(reward),
    }


def score_rollout_trajectory(
        rollout: dict,
        observation_text: str,
        kg,
        graph_samplers,
        graph_split: str = 'train') -> dict:
    history = rollout.get('history') or []
    context = str(observation_text).strip()
    action_scores = []
    for index in range(0, len(history), 2):
        action_text = str(history[index]).strip()
        if not action_text:
            continue
        action_score = score_action_completion(
            completion=action_text,
            target='',
            source=context,
            kg=kg,
            graph_split=graph_split,
        )
        action_scores.append(action_score)
        result_text = str(history[index + 1]).strip() if index + 1 < len(history) else ''
        context = '\n'.join(part for part in [context, action_text, result_text] if part)

    logic_score = score_logic_completion(
        completion=rollout.get('dsl', ''),
        source=context,
        kg=kg,
        graph_samplers=graph_samplers,
        graph_split=graph_split,
    )

    action_reward_avg = (
        sum(float(score.get('action_reward', 0.0)) for score in action_scores) / len(action_scores)
        if action_scores else 0.0
    )
    action_parse_rate = (
        sum(float(score.get('action_parse_success', 0.0)) for score in action_scores) / len(action_scores)
        if action_scores else 0.0
    )
    action_execution_rate = (
        sum(float(score.get('action_execution_success', 0.0)) for score in action_scores) / len(action_scores)
        if action_scores else 0.0
    )

    strict_score = score_action_strictness(
        history=history,
        observation_text=observation_text,
        kg=kg,
    )
    trace_score = score_trace_dsl_consistency(
        history=history,
        dsl_text=rollout.get('dsl', ''),
        observation_text=observation_text,
    )
    evidence_score = score_evidence_usage(
        history=history,
        dsl_text=rollout.get('dsl', ''),
        observation_text=observation_text,
        kg=kg,
    )
    step_score = score_step_control(
        history=history,
        dsl_text=rollout.get('dsl', ''),
    )
    bad_score = score_bad_generation(
        raw_generations=rollout.get('raw_generations', []),
        history=history,
        dsl_text=rollout.get('dsl', ''),
        kg=kg,
    )

    action_valid_scores = []
    for action_score in action_scores:
        action_valid_scores.append(
            0.4 * strict_score.get('strict_action_parse_rate', 0.0)
            + 0.4 * float(action_score.get('action_execution_success', 0.0))
            + 0.2 * float(action_score.get('target_grounding', 0.0))
        )
    action_valid = sum(action_valid_scores) / len(action_valid_scores) if action_valid_scores else 0.0
    evidence_usage_reward = (
        float(evidence_score.get('evidence_usage', 0.0))
        - 0.5 * float(evidence_score.get('hallucinated_dsl_token_rate', 0.0))
    )

    no_action_penalty = 0.5 if not action_scores else 0.0
    action_execution_error_count = sum(
        1 for score in action_scores
        if float(score.get('action_execution_success', 0.0)) <= 0.0
    )
    bad_generation_penalty = (
        float(bad_score.get('bad_generation_penalty', 0.0))
        + 1.0 * action_execution_error_count
        + 0.5 * int(strict_score.get('unknown_action_entity_count', 0))
        + 0.5 * int(evidence_score.get('unknown_dsl_entity_count', 0))
    )

    trajectory_reward = (
        float(logic_score['stage3_reward'])
        + 0.5 * action_valid
        + 0.4 * float(trace_score.get('trace_dsl_consistency', 0.0))
        + 0.3 * evidence_usage_reward
        + 0.2 * float(step_score.get('step_control', 0.0))
        + 0.1 * action_reward_avg
        - bad_generation_penalty
        - no_action_penalty
    )
    return {
        **logic_score,
        **strict_score,
        **trace_score,
        **evidence_score,
        **step_score,
        **bad_score,
        'num_actions': len(action_scores),
        'action_reward_avg': float(action_reward_avg),
        'action_valid': float(action_valid),
        'action_parse_rate': float(action_parse_rate),
        'action_execution_rate': float(action_execution_rate),
        'action_execution_error_count': int(action_execution_error_count),
        'bad_generation_penalty_total': float(bad_generation_penalty),
        'evidence_usage_reward': float(evidence_usage_reward),
        'no_action_penalty': float(no_action_penalty),
        'stage3_reward': float(trajectory_reward),
        'logic_stage3_reward': float(logic_score['stage3_reward']),
    }
