import json

from utils.condition import CONDITION_ORDER, CONDITION_LABELS, pattern_to_condition_text
from utils.parsing import qry_wordlist_2_nestedlist


OBS_PREFIX = 'OBS'
KG_HINTS_PREFIX = 'KG_HINTS'
KG_HINT_FACT_PREFIX = 'FACT'
TEXT_CONDITION_FIELD = 'condition_text_textual'
GRAPH_HYPOTHESIS_FIELD = 'hypothesis_graph_text'

LEGACY_HYPOTHESIS_STRUCTURE_TOKENS = ['(', ')', 'p', 'i', 'u', 'n', 'e']
SURFACE_HYPOTHESIS_STRUCTURE_TOKENS = ['(', ')', '[', ']', '-p', '-i', '-u', '-n', '-e']
HYPOTHESIS_STRUCTURE_TOKENS = SURFACE_HYPOTHESIS_STRUCTURE_TOKENS

GRAPH_TEXT_TOKENS = ['Conclusion', 'AND', 'OR', 'NOT', 'PROJECTION', 'ENTITY', 'leads', 'to']
KG_HINT_TOKENS = [KG_HINTS_PREFIX, KG_HINT_FACT_PREFIX]
RELATION_DIRECTION_PREFIXES = ('+', '-')
QUERY_OPERATOR_TEXT_TO_SYMBOL = {
    '-p': 'p',
    '-i': 'i',
    '-u': 'u',
    '-n': 'n',
    '-e': 'e',
}
QUERY_OPERATOR_SYMBOL_TO_TEXT = {value: key for key, value in QUERY_OPERATOR_TEXT_TO_SYMBOL.items()}
CONTROL_TEXT_TOKENS = [
    OBS_PREFIX,
    *KG_HINT_TOKENS,
    'COND',
    *CONDITION_LABELS.values(),
    *GRAPH_TEXT_TOKENS,
]


def normalize_symbol_name(name: str) -> str:
    return ' '.join(str(name).replace('_', ' ').split())


def strip_surface_brackets(text: str) -> str:
    text = str(text).strip()
    if text.startswith('[') and text.endswith(']'):
        return text[1:-1].strip()
    return text


def canonical_surface_key(text: str) -> str:
    return ' '.join(strip_surface_brackets(text).replace('_', ' ').split())


def format_surface_atom(text: str) -> str:
    return f'[{normalize_symbol_name(text)}]'


def _looks_like_surface_query(text: str) -> bool:
    text = str(text).strip()
    return '[' in text and any(operator in text for operator in QUERY_OPERATOR_TEXT_TO_SYMBOL)


def tokenize_surface_text(text: str) -> list[str]:
    tokens = []
    text = str(text)
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in '()':
            tokens.append(char)
            index += 1
            continue
        if char == '[':
            end_index = text.find(']', index + 1)
            if end_index == -1:
                raise ValueError(f'Unclosed bracket in text: {text}')
            tokens.append(text[index:end_index + 1])
            index = end_index + 1
            continue
        end_index = index
        while end_index < len(text) and (not text[end_index].isspace()) and text[end_index] not in '()[]':
            end_index += 1
        tokens.append(text[index:end_index])
        index = end_index
    return tokens


def render_surface_tokens(tokens: list[str]) -> str:
    chunks = []
    for token in tokens:
        if token == '(':
            if chunks and not chunks[-1].endswith(('(', ' ')):
                chunks.append(' ')
            chunks.append('(')
            continue
        if token == ')':
            if chunks and chunks[-1] == ' ':
                chunks.pop()
            chunks.append(')')
            continue
        if chunks and not chunks[-1].endswith(('(', ' ')):
            chunks.append(' ')
        chunks.append(token)
    return ''.join(chunks).strip()


def build_name_maps(kg):
    ent_name_to_id = {}
    rel_name_to_id = {}

    for ent_id, ent_name in kg.ent_id2name.items():
        ent_name_to_id[canonical_surface_key(ent_name)] = ent_id

    for rel_id, rel_name in kg.rel_id2name.items():
        rel_name_to_id[canonical_surface_key(rel_name)] = rel_id

    return ent_name_to_id, rel_name_to_id


def entity_id_to_text(entity_id: int, kg) -> str:
    return normalize_symbol_name(kg.ent_id2name[entity_id])


def relation_id_to_text(relation_id: int, kg) -> str:
    return normalize_symbol_name(kg.rel_id2name[-relation_id])


def get_entity_text_tokens(kg):
    return [entity_id_to_text(entity_id, kg) for entity_id in sorted(kg.ent_id2name.keys())]


def get_relation_text_tokens(kg):
    return [relation_id_to_text(-rel_id, kg) for rel_id in sorted(kg.rel_id2name.keys())]


def get_closed_text_tokens(kg):
    return list(HYPOTHESIS_STRUCTURE_TOKENS)


def entity_text_to_id(entity_text: str, kg) -> int:
    ent_name_to_id, _ = build_name_maps(kg)
    lookup_key = canonical_surface_key(entity_text)
    return ent_name_to_id[lookup_key]


def relation_text_to_id(relation_text: str, kg) -> int:
    _, rel_name_to_id = build_name_maps(kg)
    normalized = canonical_surface_key(relation_text)
    if not is_relation_text_token(normalized):
        raise ValueError(f'Invalid relation token: {relation_text}')
    return -rel_name_to_id[normalized]


def _legacy_query_text_to_wordlist(query_text: str, kg):
    wordlist = []
    for token in str(query_text).split():
        if token in LEGACY_HYPOTHESIS_STRUCTURE_TOKENS:
            wordlist.append(token)
        elif is_relation_text_token(token):
            wordlist.append(relation_text_to_id(token, kg))
        elif is_entity_text_token(token):
            wordlist.append(entity_text_to_id(token, kg))
        else:
            raise ValueError(f'Unsupported text query token: {token}')
    return wordlist


def _parse_surface_query_tokens(tokens, position, kg):
    if position >= len(tokens) or tokens[position] != '(':
        raise ValueError(f'Expected "(" at position {position}: {tokens}')

    operator_text = tokens[position + 1]
    if operator_text not in QUERY_OPERATOR_TEXT_TO_SYMBOL:
        raise ValueError(f'Unsupported query operator: {operator_text}')

    operator = QUERY_OPERATOR_TEXT_TO_SYMBOL[operator_text]
    position += 2

    if operator == 'e':
        entity_token = tokens[position]
        position += 1
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after entity token: {tokens}')
        position += 1
        return ['e', [entity_text_to_id(entity_token, kg)]], position

    if operator == 'p':
        relation_token = tokens[position]
        position += 1
        relation_id = relation_text_to_id(relation_token, kg)
        sub_query, position = _parse_surface_query_tokens(tokens, position, kg)
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after projection query: {tokens}')
        position += 1
        return ['p', [relation_id], sub_query], position

    if operator == 'n':
        sub_query, position = _parse_surface_query_tokens(tokens, position, kg)
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after negation query: {tokens}')
        position += 1
        return ['n', sub_query], position

    if operator in ['i', 'u']:
        children = []
        while position < len(tokens) and tokens[position] != ')':
            child, position = _parse_surface_query_tokens(tokens, position, kg)
            children.append(child)
        if len(children) < 2:
            raise ValueError(f'Operator "{operator}" requires at least two children: {tokens}')
        if position >= len(tokens) or tokens[position] != ')':
            raise ValueError(f'Expected ")" after set operator query: {tokens}')
        position += 1
        return [operator, *children], position

    raise ValueError(f'Unsupported operator: {operator}')


def _nested_query_to_wordlist(query_nested):
    operator, *args = query_nested

    if operator == 'e':
        entity_id = _unpack_entity_arg(args)
        return ['(', 'e', '(', entity_id, ')', ')']

    if operator == 'p':
        relation_id, sub_query = _unpack_projection_args(args)
        return ['(', 'p', '(', relation_id, ')', *_nested_query_to_wordlist(sub_query), ')']

    if operator == 'n':
        return ['(', 'n', *_nested_query_to_wordlist(args[0]), ')']

    if operator in ['i', 'u']:
        output = ['(', operator]
        for sub_query in args:
            output.extend(_nested_query_to_wordlist(sub_query))
        output.append(')')
        return output

    raise ValueError(f'Unsupported query operator: {operator}')


def query_text_to_wordlist(query_text: str, kg):
    query_text = str(query_text).strip()
    if _looks_like_surface_query(query_text):
        tokens = tokenize_surface_text(query_text)
        query_nested, position = _parse_surface_query_tokens(tokens, 0, kg)
        if position != len(tokens):
            raise ValueError(f'Unexpected trailing tokens in query text: {tokens[position:]}')
        return _nested_query_to_wordlist(query_nested)
    return _legacy_query_text_to_wordlist(query_text, kg)


def observation_text_to_answer_ids(observation_text: str, kg):
    observation_text = str(observation_text).strip()
    if '[' in observation_text and ']' in observation_text:
        tokens = tokenize_surface_text(observation_text)
        if tokens and tokens[0] == OBS_PREFIX:
            tokens = tokens[1:]
        return [
            entity_text_to_id(token, kg)
            for token in tokens
            if token.startswith('[') and token.endswith(']')
        ]

    tokens = observation_text.split()
    if tokens and tokens[0] == OBS_PREFIX:
        tokens = tokens[1:]
    return [entity_text_to_id(token, kg) for token in tokens if token]


def is_relation_text_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    normalized = canonical_surface_key(token)
    return normalized.startswith(RELATION_DIRECTION_PREFIXES)


def is_entity_text_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    normalized = canonical_surface_key(token)
    if normalized in CONTROL_TEXT_TOKENS:
        return False
    if normalized in HYPOTHESIS_STRUCTURE_TOKENS or normalized in LEGACY_HYPOTHESIS_STRUCTURE_TOKENS:
        return False
    if is_relation_text_token(normalized):
        return False
    if normalized in ['SEP', '<|pad|>', '<|endoftext|>']:
        return False
    return normalized != ''


def observation_to_text(answers, kg) -> str:
    answer_tokens = [format_surface_atom(entity_id_to_text(answer_id, kg)) for answer_id in sorted(answers)]
    return ' '.join([OBS_PREFIX, *answer_tokens])


def normalize_query_nested(query):
    if not query:
        return None
    if isinstance(query, list) and query and query[0] in ['p', 'i', 'u', 'n', 'e']:
        return query

    query_nested = qry_wordlist_2_nestedlist(list(query))
    if query_nested is None:
        raise ValueError(f'Failed to parse query wordlist: {query}')
    return query_nested


def _unpack_entity_arg(args):
    entity_value = args[0]
    if isinstance(entity_value, list):
        entity_value = entity_value[0]
    return int(entity_value)


def _unpack_projection_args(args):
    if len(args) != 2:
        raise ValueError(f'Invalid projection args: {args}')
    relation_value, sub_query = args
    if isinstance(relation_value, list):
        relation_value = relation_value[0]
    return int(relation_value), sub_query


def _nested_query_to_text_tokens(query_nested, kg):
    operator, *args = query_nested

    if operator == 'e':
        entity_id = _unpack_entity_arg(args)
        return ['(', '-e', format_surface_atom(entity_id_to_text(entity_id, kg)), ')']

    if operator == 'p':
        relation_id, sub_query = _unpack_projection_args(args)
        return [
            '(',
            '-p',
            format_surface_atom(relation_id_to_text(relation_id, kg)),
            *_nested_query_to_text_tokens(sub_query, kg),
            ')',
        ]

    if operator == 'n':
        return ['(', '-n', *_nested_query_to_text_tokens(args[0], kg), ')']

    if operator in ['i', 'u']:
        output_tokens = ['(', QUERY_OPERATOR_SYMBOL_TO_TEXT[operator]]
        for sub_query in args:
            output_tokens.extend(_nested_query_to_text_tokens(sub_query, kg))
        output_tokens.append(')')
        return output_tokens

    raise ValueError(f'Unsupported operator: {operator}')


def query_wordlist_to_text(query_wordlist, kg) -> str:
    query_nested = normalize_query_nested(query_wordlist)
    return render_surface_tokens(_nested_query_to_text_tokens(query_nested, kg))


class _GraphTextBuilder:
    def __init__(self, kg):
        self.kg = kg
        self.counters = {
            'AND': 0,
            'OR': 0,
            'NOT': 0,
            'PROJECTION': 0,
            'ENTITY': 0,
        }
        self.triples = []

    def _make_label(self, prefix: str, payload: str = '') -> str:
        node_id = self.counters[prefix]
        self.counters[prefix] += 1
        if payload:
            return f'{prefix} {payload} #{node_id}'
        return f'{prefix} #{node_id}'

    def _add_edge(self, source: str, target: str):
        self.triples.append(json.dumps([source, 'leads to', target], ensure_ascii=False))

    def _visit(self, query_nested) -> str:
        operator, *args = query_nested

        if operator == 'e':
            entity_id = _unpack_entity_arg(args)
            return self._make_label('ENTITY', entity_id_to_text(entity_id, self.kg))

        if operator == 'p':
            relation_id, sub_query = _unpack_projection_args(args)
            current = self._make_label('PROJECTION', relation_id_to_text(relation_id, self.kg))
            child = self._visit(sub_query)
            self._add_edge(current, child)
            return current

        if operator == 'n':
            current = self._make_label('NOT')
            child = self._visit(args[0])
            self._add_edge(current, child)
            return current

        if operator in ['i', 'u']:
            current = self._make_label('AND' if operator == 'i' else 'OR')
            for sub_query in args:
                child = self._visit(sub_query)
                self._add_edge(current, child)
            return current

        raise ValueError(f'Unsupported operator: {operator}')

    def build(self, query_nested) -> str:
        root = self._visit(query_nested)
        self._add_edge('Conclusion', root)
        return '\n'.join(self.triples)


def query_wordlist_to_graph_text(query_wordlist, kg) -> str:
    query_nested = normalize_query_nested(query_wordlist)
    builder = _GraphTextBuilder(kg)
    return builder.build(query_nested)


def serialize_condition_set_textual(condition_set, kg) -> str:
    if not condition_set:
        return ''

    parts = ['COND']
    for item in sorted(condition_set, key=lambda entry: CONDITION_ORDER.index(entry['type'])):
        parts.append(CONDITION_LABELS[item['type']])
        if item['type'] == 'pattern':
            parts.append(pattern_to_condition_text(item['value']))
        elif item['type'] == 'specific-entity':
            parts.append(format_surface_atom(entity_id_to_text(item['value'], kg)))
        elif item['type'] == 'specific-relation':
            parts.append(format_surface_atom(relation_id_to_text(item['value'], kg)))
        else:
            parts.append(str(item['value']))
    return ' '.join(parts)


def attach_textual_fields(record, kg, include_graph_text: bool = False):
    condition_set = []
    if record.get('condition_pattern'):
        condition_set.append({'type': 'pattern', 'value': record['condition_pattern']})
    if record.get('condition_entity_number') not in [None, -1]:
        condition_set.append({'type': 'entity-number', 'value': int(record['condition_entity_number'])})
    if record.get('condition_relation_number') not in [None, -1]:
        condition_set.append({'type': 'relation-number', 'value': int(record['condition_relation_number'])})
    if record.get('condition_specific_entity') not in [None, -1]:
        condition_set.append({'type': 'specific-entity', 'value': int(record['condition_specific_entity'])})
    if record.get('condition_specific_relation') not in [None, 0]:
        condition_set.append({'type': 'specific-relation', 'value': int(record['condition_specific_relation'])})

    enriched_record = {
        **record,
        'observation_text': observation_to_text(record['answers'], kg),
        'hypothesis_text': query_wordlist_to_text(record['query'], kg),
        TEXT_CONDITION_FIELD: serialize_condition_set_textual(condition_set, kg),
    }
    if include_graph_text:
        enriched_record[GRAPH_HYPOTHESIS_FIELD] = query_wordlist_to_graph_text(record['query'], kg)
    else:
        enriched_record.pop(GRAPH_HYPOTHESIS_FIELD, None)
    return enriched_record
