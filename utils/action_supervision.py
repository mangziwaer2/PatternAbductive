from utils.logic_dsl import parse_pattern_str
from utils.textualization import tokenize_surface_text


ACTION_CONTROL_TOKENS = [
    'ACTION',
    'TARGETS',
    'CANDIDATES',
    'OBS',
    'DIRECTION',
    'TOP_K',
    'FIND_COMMON',
    'FIND_ALTERNATIVE',
    'FIND_EXCLUSION',
    'EXPAND',
    'CHECK_COVERAGE',
]
VALID_ACTION_TYPES = {
    'FIND_COMMON',
    'FIND_ALTERNATIVE',
    'FIND_EXCLUSION',
    'EXPAND',
    'CHECK_COVERAGE',
}
ACTION_ALIASES = {
    'FIND_COMMON_CAUSE': 'FIND_COMMON',
    'FIND_ALTERNATIVE_CAUSES': 'FIND_ALTERNATIVE',
    'FIND_NEGATIVE_EVIDENCE': 'FIND_EXCLUSION',
}


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


def render_action_text(
        action_type: str,
        targets: list[str] | None = None,
        top_k: int = 10,
        candidates: list[str] | None = None,
        obs: list[str] | None = None,
        direction: str | None = None) -> str:
    action_type = normalize_action_type(action_type)
    parts = ['ACTION', action_type]

    if action_type == 'CHECK_COVERAGE':
        candidate_text = ' '.join(candidates or targets or [])
        obs_text = ' '.join(obs or [])
        parts.extend(['CANDIDATES', candidate_text, 'OBS', obs_text])
    else:
        target_text = ' '.join(targets or [])
        parts.extend(['TARGETS', target_text])
        if action_type == 'EXPAND':
            parts.extend(['DIRECTION', str(direction or 'backward')])

    parts.extend(['TOP_K', str(int(top_k))])
    return ' '.join(part for part in parts if str(part).strip()).strip()


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
