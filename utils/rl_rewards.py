from utils.action_scoring import score_action_text
from utils.execution import compute_answer_set_scores, execute_logic_text
from utils.text_scoring import score_text_query_prediction
from utils.textualization import observation_text_to_answer_ids
from utils.tool_loop import extract_action_text, extract_dsl_text


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
        target: str,
        source: str,
        condition_text: str,
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

    diagnostic = {'smatch': 0.0, 'gold_pattern_match': 0.0}
    if target:
        try:
            text_score = score_text_query_prediction(
                completion=dsl_text,
                target=target,
                source=observation_text,
                condition_text=condition_text,
                kg=kg,
                graph_samplers=graph_samplers,
                searching_split=graph_split,
            )
            diagnostic['smatch'] = float(text_score.get('smatch', 0.0))
            diagnostic['gold_pattern_match'] = float(text_score.get('validity', 0.0))
        except Exception:
            pass

    reward = (
        0.5 * execution['parse_success']
        + 0.5 * execution['execution_success']
        + 2.0 * answer_scores['answer_jaccard']
        + 1.0 * answer_scores['answer_recall']
        + 0.5 * answer_scores['answer_precision']
        + 0.2 * compactness
        - 0.1 * complexity_penalty
    )
    return {
        'target_type': 'dsl',
        **execution,
        **answer_scores,
        **diagnostic,
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
        target: str,
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
        target=target,
        source=context,
        condition_text='',
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
    no_action_penalty = 0.3 if not action_scores else 0.0
    trajectory_reward = (
        float(logic_score['stage3_reward'])
        + 0.2 * action_reward_avg
        - 0.05 * len(action_scores)
        - no_action_penalty
    )
    return {
        **logic_score,
        'num_actions': len(action_scores),
        'action_reward_avg': float(action_reward_avg),
        'action_parse_rate': float(action_parse_rate),
        'action_execution_rate': float(action_execution_rate),
        'no_action_penalty': float(no_action_penalty),
        'stage3_reward': float(trajectory_reward),
        'logic_stage3_reward': float(logic_score['stage3_reward']),
    }

