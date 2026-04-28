from utils.action_supervision import extract_observation_entity_tokens, render_action_text, VALID_ACTION_TYPES
from utils.evidence import build_common_cause_evidence, render_evidence_package
from utils.textualization import tokenize_surface_text


def parse_action_text(action_text: str) -> dict:
    tokens = tokenize_surface_text(action_text)
    if not tokens or tokens[0] != 'ACTION':
        raise ValueError(f'Action must start with ACTION: {action_text}')
    if len(tokens) < 2:
        raise ValueError(f'Missing action type: {action_text}')

    action = {
        'action_type': tokens[1],
        'targets': [],
        'top_k': 10,
    }
    if action['action_type'] not in VALID_ACTION_TYPES:
        raise ValueError(f'Unsupported action type: {action["action_type"]}')

    index = 2
    while index < len(tokens):
        token = tokens[index]
        if token == 'TARGETS':
            index += 1
            targets = []
            while index < len(tokens) and tokens[index] != 'TOP_K':
                targets.append(tokens[index])
                index += 1
            action['targets'] = targets
            continue
        if token == 'TOP_K' and index + 1 < len(tokens):
            action['top_k'] = int(tokens[index + 1])
            index += 2
            continue
        index += 1

    return action


def render_action(action: dict) -> str:
    return render_action_text(
        action_type=action.get('action_type', 'FIND_COMMON_CAUSE'),
        targets=action.get('targets', []),
        top_k=int(action.get('top_k', 10)),
    )


def execute_action(action: dict, kg, graph_split: str = 'train') -> dict:
    action_type = action.get('action_type')
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f'Unsupported action type: {action_type}')

    targets = action.get('targets') or []
    observation_text = 'OBS ' + ' '.join(targets)

    return build_common_cause_evidence(
        observation_text=observation_text,
        kg=kg,
        graph_split=graph_split,
        top_k=int(action.get('top_k', 10)),
        max_hops=1,
    )


def execute_action_text(action_text: str, kg, graph_split: str = 'train') -> str:
    action = parse_action_text(action_text)
    evidence = execute_action(action, kg=kg, graph_split=graph_split)
    return render_evidence_package(evidence, kg, prefix='RESULT')


def build_default_action_from_observation(observation_text: str, top_k: int = 10, max_hops: int = 2) -> str:
    del max_hops
    targets = extract_observation_entity_tokens(observation_text)
    return render_action({
        'action_type': 'FIND_COMMON_CAUSE',
        'targets': targets,
        'top_k': top_k,
    })
