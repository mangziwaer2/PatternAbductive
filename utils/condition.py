import random
from itertools import combinations

from utils.parsing import qry_str_2_wordlist, qry_wordlist_2_nestedlist


CONDITION_ORDER = (
    'pattern',
    'entity-number',
    'relation-number',
    'specific-entity',
    'specific-relation',
)

CONDITION_LABELS = {
    'pattern': 'PATTERN',
    'entity-number': 'ENT_NUM',
    'relation-number': 'REL_NUM',
    'specific-entity': 'SPEC_ENT',
    'specific-relation': 'SPEC_REL',
}

CONDITION_FIELD_NAMES = {
    'pattern': 'condition_pattern',
    'entity-number': 'condition_entity_number',
    'relation-number': 'condition_relation_number',
    'specific-entity': 'condition_specific_entity',
    'specific-relation': 'condition_specific_relation',
}

CONDITION_TOKENS = ['COND', *CONDITION_LABELS.values()]
DEFAULT_EXCLUDED_CONDITION_TYPES = {'entity-number', 'relation-number'}


def normalize_query_wordlist(query):
    if isinstance(query, str):
        return qry_str_2_wordlist(query)
    return list(query)


def _infer_pattern_from_nested(query_nested):
    operator, *args = query_nested

    if operator == 'e':
        return '(e)'
    if operator == 'p':
        return f'(p,{_infer_pattern_from_nested(args[1])})'
    if operator == 'n':
        return f'(n,{_infer_pattern_from_nested(args[0])})'
    if operator in ['i', 'u']:
        return f'({operator},{",".join(_infer_pattern_from_nested(arg) for arg in args)})'

    raise ValueError(f'Unsupported operator: {operator}')


def _collect_query_atoms(query_nested):
    operator, *args = query_nested

    if operator == 'e':
        [[entity_id]] = args
        return [entity_id], []
    if operator == 'p':
        [relation_id], sub_query = args
        entities, relations = _collect_query_atoms(sub_query)
        return entities, [relation_id] + relations
    if operator == 'n':
        return _collect_query_atoms(args[0])
    if operator in ['i', 'u']:
        entities = []
        relations = []
        for sub_query in args:
            sub_entities, sub_relations = _collect_query_atoms(sub_query)
            entities.extend(sub_entities)
            relations.extend(sub_relations)
        return entities, relations

    raise ValueError(f'Unsupported operator: {operator}')


def pattern_to_condition_text(pattern_str):
    condition_text = pattern_str.replace(',', ' ')
    condition_text = condition_text.replace('(', ' ( ')
    condition_text = condition_text.replace(')', ' ) ')
    return ' '.join(condition_text.split())


def extract_condition_metadata(query, pattern_str=None):
    query_wordlist = normalize_query_wordlist(query)
    query_nested = qry_wordlist_2_nestedlist(query_wordlist)
    if query_nested is None:
        raise ValueError(f'Failed to parse query: {query_wordlist}')

    entities, relations = _collect_query_atoms(query_nested)
    unique_entities = sorted(set(entities))
    unique_relations = sorted(set(relations))

    if pattern_str is None:
        pattern_str = _infer_pattern_from_nested(query_nested)

    return {
        'pattern': pattern_str,
        'pattern_condition_text': pattern_to_condition_text(pattern_str),
        # Count occurrences instead of unique ids so the condition matches
        # the structure length of the logical hypothesis.
        'entity_number': len(entities),
        'relation_number': len(relations),
        'unique_entity_number': len(unique_entities),
        'unique_relation_number': len(unique_relations),
        'anchor_entities': entities,
        'unique_anchor_entities': unique_entities,
        'relations': relations,
        'unique_relations': unique_relations,
    }


def _choose_condition_value(metadata, condition_type, rng):
    if condition_type == 'pattern':
        return metadata['pattern']
    if condition_type == 'entity-number':
        return metadata['entity_number']
    if condition_type == 'relation-number':
        return metadata['relation_number']
    if condition_type == 'specific-entity':
        return rng.choice(metadata['unique_anchor_entities'])
    if condition_type == 'specific-relation':
        return rng.choice(metadata['unique_relations'])
    raise ValueError(f'Unsupported condition type: {condition_type}')


def normalize_condition_type_list(condition_types) -> list[str]:
    if condition_types is None:
        return []
    if isinstance(condition_types, str):
        return [
            token.strip()
            for token in condition_types.split(',')
            if token.strip() != ''
        ]
    return [str(token).strip() for token in condition_types if str(token).strip() != '']


def get_available_condition_types(metadata, excluded_types=None):
    excluded_types = set(normalize_condition_type_list(excluded_types))
    condition_types = []
    if metadata['pattern']:
        condition_types.append('pattern')
    if metadata['entity_number'] > 0:
        condition_types.append('entity-number')
    if metadata['relation_number'] > 0:
        condition_types.append('relation-number')
    if metadata['unique_anchor_entities']:
        condition_types.append('specific-entity')
    if metadata['unique_relations']:
        condition_types.append('specific-relation')
    return [condition_type for condition_type in condition_types if condition_type not in excluded_types]


def build_condition_set(condition_types, metadata, rng=None):
    if rng is None:
        rng = random

    ordered_types = sorted(condition_types, key=CONDITION_ORDER.index)
    return [
        {
            'type': condition_type,
            'value': _choose_condition_value(metadata, condition_type, rng),
        }
        for condition_type in ordered_types
    ]


def serialize_condition_set(condition_set):
    if not condition_set:
        return ''

    parts = ['COND']
    ordered_items = sorted(condition_set, key=lambda item: CONDITION_ORDER.index(item['type']))
    for item in ordered_items:
        label = CONDITION_LABELS[item['type']]
        value = item['value']
        if item['type'] == 'pattern':
            value_text = pattern_to_condition_text(value)
        else:
            value_text = str(value)
        parts.extend([label, value_text])
    return ' '.join(parts)


def condition_set_key(condition_set):
    ordered_items = sorted(condition_set, key=lambda item: CONDITION_ORDER.index(item['type']))
    return tuple((item['type'], item['value']) for item in ordered_items)


def flatten_condition_set(condition_set):
    flat = {
        'condition_text': serialize_condition_set(condition_set),
        'condition_types': [item['type'] for item in sorted(condition_set, key=lambda item: CONDITION_ORDER.index(item['type']))],
        'condition_signature': 'unconditional',
        'condition_size': len(condition_set),
    }
    for field_name in CONDITION_FIELD_NAMES.values():
        flat[field_name] = None

    if condition_set:
        flat['condition_signature'] = '+'.join(flat['condition_types'])

    for item in condition_set:
        flat[CONDITION_FIELD_NAMES[item['type']]] = item['value']

    return flat


def sample_condition_sets(metadata, samples_per_query, max_condition_arity=3, rng=None, excluded_types=None):
    if rng is None:
        rng = random
    if samples_per_query <= 0:
        return []

    available_types = get_available_condition_types(metadata, excluded_types=excluded_types)
    if not available_types:
        return []

    max_condition_arity = max(1, min(max_condition_arity, len(available_types)))

    condition_sets = []
    seen = set()

    def maybe_add(condition_types):
        condition_set = build_condition_set(condition_types, metadata, rng=rng)
        key = condition_set_key(condition_set)
        if key in seen:
            return
        seen.add(key)
        condition_sets.append(condition_set)

    for condition_type in available_types:
        maybe_add([condition_type])
        if len(condition_sets) >= samples_per_query:
            return condition_sets

    type_combinations = []
    for arity in range(2, max_condition_arity + 1):
        type_combinations.extend(combinations(available_types, arity))
    rng.shuffle(type_combinations)

    for condition_types in type_combinations:
        maybe_add(condition_types)
        if len(condition_sets) >= samples_per_query:
            return condition_sets

    attempts = 0
    max_attempts = samples_per_query * 10
    while len(condition_sets) < samples_per_query and attempts < max_attempts:
        arity = rng.randint(1, max_condition_arity)
        condition_types = rng.sample(available_types, arity)
        maybe_add(condition_types)
        attempts += 1

    return condition_sets


def expand_sample_with_conditions(
        base_record,
        samples_per_query,
        max_condition_arity=3,
        include_unconditional=True,
        rng=None,
        excluded_condition_types=None):
    if rng is None:
        rng = random

    metadata = extract_condition_metadata(
        query=base_record['query'],
        pattern_str=base_record.get('pattern_str'),
    )

    shared_fields = {
        **base_record,
        'query_entity_number': metadata['entity_number'],
        'query_relation_number': metadata['relation_number'],
        'query_unique_entity_number': metadata['unique_entity_number'],
        'query_unique_relation_number': metadata['unique_relation_number'],
        'query_anchor_entities': metadata['unique_anchor_entities'],
        'query_relations': metadata['unique_relations'],
    }

    records = []
    if include_unconditional:
        records.append({
            **shared_fields,
            **flatten_condition_set([]),
        })

    excluded_condition_types = normalize_condition_type_list(excluded_condition_types)
    for condition_set in sample_condition_sets(
            metadata=metadata,
            samples_per_query=samples_per_query,
            max_condition_arity=max_condition_arity,
            rng=rng,
            excluded_types=excluded_condition_types):
        records.append({
            **shared_fields,
            **flatten_condition_set(condition_set),
        })

    return records
