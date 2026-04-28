from utils.textualization import (
    entity_id_to_text,
    entity_text_to_id,
    format_surface_atom,
    normalize_query_nested,
    relation_id_to_text,
    relation_text_to_id,
    query_text_to_wordlist,
)


DSL_CONTROL_TOKENS = [
    'PATTERN',
    'DSL',
    'QUERY',
    'ENT',
    'PROJ',
    'AND',
    'OR',
    'NOT',
]

DSL_OPERATOR_TO_SYMBOL = {
    'ENT': 'e',
    'PROJ': 'p',
    'AND': 'i',
    'OR': 'u',
    'NOT': 'n',
}

SYMBOL_TO_DSL_OPERATOR = {
    'e': 'ENT',
    'p': 'PROJ',
    'i': 'AND',
    'u': 'OR',
    'n': 'NOT',
}


def _tokenize_pattern(pattern_str: str) -> list[str]:
    tokens = []
    index = 0
    text = str(pattern_str).strip()
    while index < len(text):
        char = text[index]
        if char.isspace() or char == ',':
            index += 1
            continue
        if char in '()':
            tokens.append(char)
            index += 1
            continue
        if char in {'p', 'i', 'u', 'n', 'e'}:
            tokens.append(char)
            index += 1
            continue
        raise ValueError(f'Unsupported pattern token at index {index}: {pattern_str}')
    return tokens


def _parse_pattern_tokens(tokens: list[str], position: int = 0):
    if position >= len(tokens) or tokens[position] != '(':
        raise ValueError(f'Expected "(" at position {position}: {tokens}')
    if position + 1 >= len(tokens):
        raise ValueError(f'Missing pattern operator after "(": {tokens}')

    operator = tokens[position + 1]
    position += 2

    if operator == 'e':
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after entity pattern: {tokens}')
        return ['e'], position + 1

    if operator in {'p', 'n'}:
        child, position = _parse_pattern_tokens(tokens, position)
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after unary pattern: {tokens}')
        return [operator, child], position + 1

    if operator in {'i', 'u'}:
        children = []
        while position < len(tokens) and tokens[position] != ')':
            child, position = _parse_pattern_tokens(tokens, position)
            children.append(child)
        if len(children) < 2:
            raise ValueError(f'Pattern operator "{operator}" requires at least two children: {tokens}')
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after set pattern: {tokens}')
        return [operator, *children], position + 1

    raise ValueError(f'Unsupported pattern operator: {operator}')


def parse_pattern_str(pattern_str: str):
    tokens = _tokenize_pattern(pattern_str)
    pattern, position = _parse_pattern_tokens(tokens, 0)
    if position != len(tokens):
        raise ValueError(f'Unexpected trailing pattern tokens: {tokens[position:]}')
    return pattern


def _render_pattern_dsl(pattern_nested) -> str:
    operator, *args = pattern_nested
    if operator == 'e':
        return 'ENT'
    if operator == 'p':
        return f'PROJ({_render_pattern_dsl(args[0])})'
    if operator == 'n':
        return f'NOT({_render_pattern_dsl(args[0])})'
    if operator in {'i', 'u'}:
        name = 'AND' if operator == 'i' else 'OR'
        return f'{name}(' + ', '.join(_render_pattern_dsl(child) for child in args) + ')'
    raise ValueError(f'Unsupported pattern operator: {operator}')


def pattern_str_to_pattern_dsl(pattern_str: str) -> str:
    return _render_pattern_dsl(parse_pattern_str(pattern_str))


def _render_query_dsl(query_nested, kg) -> str:
    operator, *args = query_nested

    if operator == 'e':
        entity_value = args[0]
        if isinstance(entity_value, list):
            entity_value = entity_value[0]
        return f'ENT({format_surface_atom(entity_id_to_text(int(entity_value), kg))})'

    if operator == 'p':
        relation_value, sub_query = args
        if isinstance(relation_value, list):
            relation_value = relation_value[0]
        relation_text = format_surface_atom(relation_id_to_text(int(relation_value), kg))
        return f'PROJ({relation_text}, {_render_query_dsl(sub_query, kg)})'

    if operator == 'n':
        return f'NOT({_render_query_dsl(args[0], kg)})'

    if operator in {'i', 'u'}:
        name = 'AND' if operator == 'i' else 'OR'
        return f'{name}(' + ', '.join(_render_query_dsl(child, kg) for child in args) + ')'

    raise ValueError(f'Unsupported query operator: {operator}')


def query_wordlist_to_dsl(query_wordlist, kg) -> str:
    query_nested = normalize_query_nested(query_wordlist)
    return _render_query_dsl(query_nested, kg)


def surface_query_to_dsl(query_text: str, kg) -> str:
    return query_wordlist_to_dsl(query_text_to_wordlist(query_text, kg), kg)


def _tokenize_dsl(text: str) -> list[str]:
    tokens = []
    index = 0
    text = str(text).strip()
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in '(),':
            tokens.append(char)
            index += 1
            continue
        if char == '[':
            end_index = text.find(']', index + 1)
            if end_index == -1:
                raise ValueError(f'Unclosed bracket in DSL: {text}')
            tokens.append(text[index:end_index + 1])
            index = end_index + 1
            continue
        end_index = index
        while end_index < len(text) and not text[end_index].isspace() and text[end_index] not in '(),[]':
            end_index += 1
        tokens.append(text[index:end_index])
        index = end_index
    return tokens


def _strip_dsl_prefix(text: str) -> str:
    text = str(text).strip()
    if not text:
        return text
    for marker in ['DSL', 'QUERY']:
        if text.startswith(marker):
            return text[len(marker):].strip()
    if ' DSL ' in text:
        return text.split(' DSL ', 1)[1].strip()
    if '\nDSL\n' in text:
        return text.split('\nDSL\n', 1)[1].strip()
    return text


def extract_dsl_text(text: str) -> str:
    return _strip_dsl_prefix(text)


def _parse_dsl_expr(tokens: list[str], position: int, kg):
    if position >= len(tokens):
        raise ValueError('Unexpected end of DSL tokens')

    operator = tokens[position]
    if operator not in DSL_OPERATOR_TO_SYMBOL:
        raise ValueError(f'Unsupported DSL operator: {operator}')
    position += 1

    if position >= len(tokens) or tokens[position] != '(':
        raise ValueError(f'Expected "(" after {operator}: {tokens}')
    position += 1

    if operator == 'ENT':
        if position >= len(tokens):
            raise ValueError(f'Missing ENT payload: {tokens}')
        entity_token = tokens[position]
        position += 1
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after ENT payload: {tokens}')
        return ['e', [entity_text_to_id(entity_token, kg)]], position + 1

    if operator == 'PROJ':
        if position >= len(tokens):
            raise ValueError(f'Missing PROJ relation payload: {tokens}')
        relation_token = tokens[position]
        position += 1
        if position >= len(tokens) or tokens[position] != ',':
            raise ValueError(f'Expected "," after PROJ relation: {tokens}')
        position += 1
        sub_query, position = _parse_dsl_expr(tokens, position, kg)
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after PROJ query: {tokens}')
        return ['p', [relation_text_to_id(relation_token, kg)], sub_query], position + 1

    if operator == 'NOT':
        sub_query, position = _parse_dsl_expr(tokens, position, kg)
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after NOT query: {tokens}')
        return ['n', sub_query], position + 1

    if operator in {'AND', 'OR'}:
        children = []
        while True:
            child, position = _parse_dsl_expr(tokens, position, kg)
            children.append(child)
            if position >= len(tokens):
                raise ValueError(f'Unclosed {operator} expression: {tokens}')
            if tokens[position] == ',':
                position += 1
                continue
            if tokens[position] == ')':
                break
            raise ValueError(f'Expected "," or ")" in {operator}: {tokens}')
        if len(children) < 2:
            raise ValueError(f'{operator} requires at least two children: {tokens}')
        return [DSL_OPERATOR_TO_SYMBOL[operator], *children], position + 1

    raise ValueError(f'Unsupported DSL operator: {operator}')


def _nested_query_to_wordlist(query_nested):
    operator, *args = query_nested

    if operator == 'e':
        entity_value = args[0]
        if isinstance(entity_value, list):
            entity_value = entity_value[0]
        return ['(', 'e', '(', int(entity_value), ')', ')']

    if operator == 'p':
        relation_value, sub_query = args
        if isinstance(relation_value, list):
            relation_value = relation_value[0]
        return ['(', 'p', '(', int(relation_value), ')', *_nested_query_to_wordlist(sub_query), ')']

    if operator == 'n':
        return ['(', 'n', *_nested_query_to_wordlist(args[0]), ')']

    if operator in {'i', 'u'}:
        output = ['(', operator]
        for sub_query in args:
            output.extend(_nested_query_to_wordlist(sub_query))
        output.append(')')
        return output

    raise ValueError(f'Unsupported query operator: {operator}')


def dsl_to_query_wordlist(text: str, kg):
    dsl_text = extract_dsl_text(text)
    tokens = _tokenize_dsl(dsl_text)
    query_nested, position = _parse_dsl_expr(tokens, 0, kg)
    if position != len(tokens):
        raise ValueError(f'Unexpected trailing DSL tokens: {tokens[position:]}')
    return _nested_query_to_wordlist(query_nested)


def looks_like_dsl(text: str) -> bool:
    stripped = extract_dsl_text(text)
    return any(stripped.startswith(f'{operator}(') for operator in ['ENT', 'PROJ', 'AND', 'OR', 'NOT'])


def logic_text_to_query_wordlist(text: str, kg):
    if looks_like_dsl(text):
        return dsl_to_query_wordlist(text, kg)
    return query_text_to_wordlist(text, kg)


def build_pattern_dsl_target(pattern_str: str, query_text: str, kg) -> str:
    return f'PATTERN {pattern_str_to_pattern_dsl(pattern_str)} DSL {surface_query_to_dsl(query_text, kg)}'
