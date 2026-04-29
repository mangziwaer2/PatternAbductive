from utils.logic_dsl import parse_pattern_str
from utils.textualization import tokenize_surface_text


ACTION_CONTROL_TOKENS = [
    '<ACTION>',
    '</ACTION>',
    'ACTION',
    'TARGETS',
    'DIRECTION',
    'TOP_K',
    'FIND_COMMON',
    'FIND_ALTERNATIVE',
    'FIND_EXCLUSION',
    'EXPAND',
]
VALID_ACTION_TYPES = {
    'FIND_COMMON',
    'FIND_ALTERNATIVE',
    'FIND_EXCLUSION',
    'EXPAND',
}
ACTION_ALIASES = {
    'FIND_COMMON_CAUSE': 'FIND_COMMON',
    'FIND_ALTERNATIVE_CAUSES': 'FIND_ALTERNATIVE',
    'FIND_NEGATIVE_EVIDENCE': 'FIND_EXCLUSION',
}
ACTION_START_TAG = '<ACTION>'
ACTION_END_TAG = '</ACTION>'


def extract_observation_entity_tokens(observation_text: str) -> list[str]:
    tokens = tokenize_surface_text(observation_text)
    if tokens and tokens[0] == 'OBS':
        tokens = tokens[1:]
    return [
        token
        for token in tokens
        if isinstance(token, str) and token.startswith('[') and token.endswith(']')
    ]


def _walk_pattern(pattern_nested):
    operator, *args = pattern_nested
    yield operator, args
    for arg in args:
        if isinstance(arg, list):
            yield from _walk_pattern(arg)


def _max_projection_depth(pattern_nested) -> int:
    operator, *args = pattern_nested
    if operator == 'p':
        return 1 + _max_projection_depth(args[0])
    if not args:
        return 0
    return max((_max_projection_depth(arg) for arg in args if isinstance(arg, list)), default=0)


def _projection_leaf_depths(pattern_nested) -> list[int]:
    operator, *args = pattern_nested
    if operator == 'e':
        return [0]
    if operator == 'p':
        return [depth + 1 for depth in _projection_leaf_depths(args[0])]
    if operator == 'n':
        return _projection_leaf_depths(args[0])
    if operator in {'i', 'u'}:
        depths = []
        for arg in args:
            depths.extend(_projection_leaf_depths(arg))
        return depths
    return [0]


def infer_branch_hops(pattern_str: str) -> list[int]:
    depths = _projection_leaf_depths(parse_pattern_str(pattern_str))
    return [max(1, int(depth)) for depth in depths] or [1]


def infer_action_type(pattern_str: str) -> str:
    pattern_nested = parse_pattern_str(pattern_str)
    operators = [operator for operator, _ in _walk_pattern(pattern_nested)]

    if 'n' in operators:
        return 'FIND_EXCLUSION'
    if 'u' in operators:
        return 'FIND_ALTERNATIVE'
    return 'FIND_COMMON'


def infer_max_hops(pattern_str: str, minimum: int = 1, maximum: int = 4) -> int:
    depth = max(infer_branch_hops(pattern_str), default=_max_projection_depth(parse_pattern_str(pattern_str)))
    return max(minimum, min(maximum, depth if depth > 0 else minimum))


def infer_action_steps(pattern_str: str, minimum: int = 1, maximum: int = 4) -> int:
    return infer_max_hops(pattern_str=pattern_str, minimum=minimum, maximum=maximum)


def normalize_action_type(action_type: str) -> str:
    normalized = ACTION_ALIASES.get(str(action_type).strip(), str(action_type).strip())
    return normalized


def strip_action_tags(action_text: str) -> str:
    text = str(action_text or '').strip()
    start_index = text.find(ACTION_START_TAG)
    if start_index >= 0:
        text = text[start_index + len(ACTION_START_TAG):]
        end_index = text.find(ACTION_END_TAG)
        if end_index >= 0:
            text = text[:end_index]
        return text.strip()
    end_index = text.find(ACTION_END_TAG)
    if end_index >= 0:
        text = text[:end_index]
    return text.strip()


def action_has_explicit_boundary(action_text: str) -> bool:
    text = str(action_text or '')
    return ACTION_START_TAG in text and ACTION_END_TAG in text


def tag_action_text(action_text: str) -> str:
    body = strip_action_tags(action_text)
    return '\n'.join([ACTION_START_TAG, body, ACTION_END_TAG])


def render_action_text(
        action_type: str,
        targets: list[str] | None = None,
        top_k: int = 10,
        direction: str | None = None,
        tagged: bool = True) -> str:
    action_type = normalize_action_type(action_type)
    parts = ['ACTION', action_type]

    target_text = ' '.join(targets or [])
    parts.extend(['TARGETS', target_text])
    if action_type == 'EXPAND':
        parts.extend(['DIRECTION', str(direction or 'backward')])

    parts.extend(['TOP_K', str(int(top_k))])
    body = ' '.join(part for part in parts if str(part).strip()).strip()
    return tag_action_text(body) if tagged else body


def build_action_text(
        pattern_str: str,
        observation_text: str,
        top_k: int = 10,
        max_hops: int | None = None) -> str:
    del max_hops
    action_type = infer_action_type(pattern_str)
    targets = extract_observation_entity_tokens(observation_text)
    return render_action_text(action_type=action_type, targets=targets, top_k=top_k)


def build_action_text_for_record(record, top_k: int = 10) -> str:
    return build_action_text(
        pattern_str=record['pattern_str'],
        observation_text=record['observation_text'],
        top_k=top_k,
    )
