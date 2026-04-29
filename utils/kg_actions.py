from utils.action_supervision import (
    extract_observation_entity_tokens,
    normalize_action_type,
    render_action_text,
    strip_action_tags,
    VALID_ACTION_TYPES,
)
from utils.evidence import (
    build_common_cause_evidence,
    build_expand_evidence,
    render_evidence_package,
)
from utils.textualization import tokenize_surface_text


ACTION_FIELD_TOKENS = {'TARGETS', 'DIRECTION', 'TOP_K'}


def parse_action_text(action_text: str) -> dict:
    tokens = tokenize_surface_text(strip_action_tags(action_text))
    if not tokens or tokens[0] != 'ACTION':
        raise ValueError(f'Action must start with ACTION: {action_text}')
    if len(tokens) < 2:
        raise ValueError(f'Missing action type: {action_text}')

    action = {
        'action_type': normalize_action_type(tokens[1]),
        'targets': [],
        'direction': 'backward',
        'top_k': 10,
    }
    if action['action_type'] not in VALID_ACTION_TYPES:
        raise ValueError(f'Unsupported action type: {action["action_type"]}')

    index = 2
    while index < len(tokens):
        token = tokens[index]
        if token == 'TARGETS':
            index += 1
            values = []
            while index < len(tokens) and tokens[index] not in ACTION_FIELD_TOKENS:
                values.append(tokens[index])
                index += 1
            action['targets'] = values
            continue
        if token == 'DIRECTION' and index + 1 < len(tokens):
            action['direction'] = str(tokens[index + 1]).lower()
            index += 2
            continue
        if token == 'TOP_K' and index + 1 < len(tokens):
            action['top_k'] = int(tokens[index + 1])
            index += 2
            continue
        index += 1

    return action


def render_action(action: dict) -> str:
    return render_action_text(
        action_type=action.get('action_type', 'FIND_COMMON'),
        targets=action.get('targets', []),
        direction=action.get('direction', 'backward'),
        top_k=int(action.get('top_k', 10)),
    )


def execute_action(action: dict, kg, graph_split: str = 'train') -> dict:
    action_type = normalize_action_type(action.get('action_type'))
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f'Unsupported action type: {action_type}')

    top_k = int(action.get('top_k', 10))

    if action_type in {'FIND_COMMON', 'FIND_ALTERNATIVE', 'FIND_EXCLUSION'}:
        targets = action.get('targets') or []
        observation_text = 'OBS ' + ' '.join(targets)
        # Fetch extra candidates for alternative/exclusion actions, then let
        # rendering keep the requested top_k after action-specific ordering.
        evidence = build_common_cause_evidence(
            observation_text=observation_text,
            kg=kg,
            graph_split=graph_split,
            top_k=top_k * 3 if action_type != 'FIND_COMMON' else top_k,
            max_hops=1,
        )
        if action_type == 'FIND_ALTERNATIVE':
            evidence['mode'] = 'find_alternative'
            evidence['candidates'] = sorted(
                evidence.get('candidates', []),
                key=lambda item: (-item.get('coverage_num', 0), item.get('entity_id', 0)),
            )[:top_k]
        elif action_type == 'FIND_EXCLUSION':
            evidence['mode'] = 'find_exclusion'
            evidence['candidates'] = sorted(
                evidence.get('candidates', []),
                key=lambda item: (
                    len(item.get('missing_ids', [])) == 0,
                    -item.get('coverage_num', 0),
                    len(item.get('missing_ids', [])),
                    item.get('entity_id', 0),
                ),
            )[:top_k]
        else:
            evidence['mode'] = 'find_common'
        return evidence

    if action_type == 'EXPAND':
        targets = action.get('targets') or []
        observation_text = 'OBS ' + ' '.join(targets)
        return build_expand_evidence(
            target_text=observation_text,
            kg=kg,
            graph_split=graph_split,
            top_k=top_k,
            direction=action.get('direction', 'backward'),
        )

    raise ValueError(f'Unsupported action type: {action_type}')


def execute_action_text(action_text: str, kg, graph_split: str = 'train') -> str:
    action = parse_action_text(action_text)
    evidence = execute_action(action, kg=kg, graph_split=graph_split)
    return render_evidence_package(evidence, kg, prefix='RESULT')


def build_default_action_from_observation(observation_text: str, top_k: int = 10, max_hops: int = 2) -> str:
    del max_hops
    targets = extract_observation_entity_tokens(observation_text)
    return render_action({
        'action_type': 'FIND_COMMON',
        'targets': targets,
        'top_k': top_k,
    })
