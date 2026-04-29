from utils.action_supervision import strip_action_tags
from utils.kg_actions import execute_action, parse_action_text
from utils.evidence import strip_result_tags
from utils.textualization import tokenize_surface_text


def _dedupe_keep_order(values):
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def compute_evidence_coverage(evidence: dict) -> float:
    candidates = evidence.get('candidates', [])
    if not candidates:
        return 0.0
    best = max(
        (
            candidate.get('coverage_num', 0) / max(candidate.get('coverage_den', 1), 1)
            for candidate in candidates
        ),
        default=0.0,
    )
    return float(best)


def extract_frontier_entity_tokens(context_text: str) -> list[str]:
    candidate_tokens = []
    observation_tokens = []
    for line in strip_result_tags(context_text).splitlines():
        if line == 'RESULT':
            continue
        tokens = tokenize_surface_text(line)
        if not tokens:
            continue
        if len(tokens) >= 5 and tokens[1] == '--' and tokens[3] == '-->':
            candidate_tokens.append(tokens[0])
        elif tokens[0] == 'OBS':
            observation_tokens.extend(
                token
                for token in tokens[1:]
                if isinstance(token, str) and token.startswith('[') and token.endswith(']')
            )
    return _dedupe_keep_order(candidate_tokens or observation_tokens)


def compute_target_grounding(action_targets: list[str], frontier_targets: list[str]) -> dict:
    action_target_set = set(action_targets)
    frontier_set = set(frontier_targets)
    if not action_targets:
        return {
            'target_grounding': 0.0,
            'target_frontier_recall': 0.0,
            'target_diversity': 0.0,
        }
    overlap = len(action_target_set & frontier_set)
    return {
        'target_grounding': overlap / max(len(action_target_set), 1),
        'target_frontier_recall': overlap / max(len(frontier_set), 1),
        'target_diversity': len(action_target_set) / max(len(action_targets), 1),
    }


def score_action_text(
        action_text: str,
        observation_text: str,
        kg,
        graph_split: str = 'train') -> dict:
    result = {
        'action_parse_success': 0.0,
        'action_execution_success': 0.0,
        'evidence_coverage': 0.0,
        'candidate_count': 0,
        'candidate_count_score': 0.0,
        'target_grounding': 0.0,
        'target_frontier_recall': 0.0,
        'target_diversity': 0.0,
        'repeat_action_penalty': 0.0,
        'action_reward': 0.0,
        'error': '',
    }

    try:
        action = parse_action_text(action_text)
    except Exception as exc:
        result['error'] = f'action_parse_error: {exc}'
        return result

    result['action_parse_success'] = 1.0
    action_targets = action.get('targets') or action.get('candidates') or action.get('obs') or []
    grounding = compute_target_grounding(
        action_targets=action_targets,
        frontier_targets=extract_frontier_entity_tokens(observation_text),
    )
    result.update(grounding)

    try:
        evidence = execute_action(action, kg=kg, graph_split=graph_split)
    except Exception as exc:
        result['error'] = f'action_execution_error: {exc}'
        return result

    result['action_execution_success'] = 1.0
    result['candidate_count'] = len(evidence.get('candidates', []))
    result['candidate_count_score'] = min(result['candidate_count'], int(action.get('top_k', 10))) / max(int(action.get('top_k', 10)), 1)
    result['evidence_coverage'] = compute_evidence_coverage(evidence)
    current_action = strip_action_tags(action_text)
    previous_actions = {
        strip_action_tags(line.strip())
        for line in str(observation_text).splitlines()
        if line.strip().startswith('ACTION ') or line.strip().startswith('<ACTION>')
    }
    result['repeat_action_penalty'] = 1.0 if current_action in previous_actions else 0.0

    result['action_reward'] = float(
        0.2 * result['action_parse_success']
        + 0.4 * result['action_execution_success']
        + 0.2 * result['evidence_coverage']
        + 0.4 * result['candidate_count_score']
        + 0.4 * result['target_grounding']
        + 0.1 * result['target_diversity']
        - 0.5 * result['repeat_action_penalty']
    )
    return result
