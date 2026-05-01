from collections import defaultdict
import random

from utils.textualization import (
    entity_id_to_text,
    format_surface_atom,
    normalize_symbol_name,
    observation_text_to_answer_ids,
)


EVIDENCE_CONTROL_TOKENS = [
    '<RESULT>',
    '</RESULT>',
    # Kept for backward-compatible tokenizer sizes with older checkpoints.
    'RESULT',
    'EVIDENCE',
]
RESULT_START_TAG = '<RESULT>'
RESULT_END_TAG = '</RESULT>'


def _drop_result_prefix(text: str) -> str:
    lines = str(text or '').strip().splitlines()
    while lines and lines[0].strip() == 'RESULT':
        lines = lines[1:]
    return '\n'.join(lines).strip()


def strip_result_tags(result_text: str) -> str:
    text = str(result_text or '').strip()
    start_index = text.find(RESULT_START_TAG)
    if start_index >= 0:
        text = text[start_index + len(RESULT_START_TAG):]
        end_index = text.find(RESULT_END_TAG)
        if end_index >= 0:
            text = text[:end_index]
        return _drop_result_prefix(text)
    end_index = text.find(RESULT_END_TAG)
    if end_index >= 0:
        text = text[:end_index]
    return _drop_result_prefix(text)


def tag_result_text(result_text: str) -> str:
    body = strip_result_tags(result_text)
    return '\n'.join([RESULT_START_TAG, body, RESULT_END_TAG])


def _entity_text(entity_id: int, kg) -> str:
    return format_surface_atom(entity_id_to_text(int(entity_id), kg))


def _relation_text(relation_id: int, kg) -> str:
    return format_surface_atom(normalize_symbol_name(kg.rel_id2name[int(relation_id)]))


def _resolve_graph_sampler(kg, graph_split: str):
    split = graph_split if graph_split in kg.graph_samplers else 'train'
    return kg.graph_samplers[split]


def _format_edge(edge: dict, kg) -> str:
    return (
        f'{_entity_text(edge["subject_id"], kg)} '
        f'--{_relation_text(edge["relation_id"], kg)}--> '
        f'{_entity_text(edge["object_id"], kg)}'
    )


def _is_self_loop(edge: dict) -> bool:
    return int(edge['subject_id']) == int(edge['object_id'])


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for edge in edges:
        if _is_self_loop(edge):
            continue
        key = (int(edge['subject_id']), int(edge['relation_id']), int(edge['object_id']))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _sample_items(items: list, k: int) -> list:
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    return random.sample(list(items), k)


def _candidate_from_edge(edge: dict, coverage_den: int = 1) -> dict:
    return {
        'entity_id': int(edge['subject_id']),
        'support': [edge],
        'missing_ids': [],
        'coverage_num': 1,
        'coverage_den': max(int(coverage_den), 1),
        'score': 1.0 / max(int(coverage_den), 1),
    }


def build_common_cause_evidence(
        observation_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10,
        max_hops: int = 2,
        max_supports_per_candidate: int = 4):
    del max_hops  # Reserved for future multi-hop evidence builders.

    observation_ids = observation_text_to_answer_ids(observation_text, kg)
    sampler = _resolve_graph_sampler(kg, graph_split)
    support_by_candidate = defaultdict(list)

    for obs_index, obs_id in enumerate(observation_ids):
        for source_id, _, relation_id in sampler.in_edges(obs_id):
            if int(source_id) == int(obs_id):
                continue
            support_by_candidate[int(source_id)].append({
                'subject_id': int(source_id),
                'relation_id': int(relation_id),
                'object_id': int(obs_id),
                'observation_index': obs_index,
            })

    candidates = []
    observation_id_set = set(observation_ids)
    for candidate_id, supports in support_by_candidate.items():
        supports = _dedupe_edges(supports)
        if not supports:
            continue
        covered_ids = {support['object_id'] for support in supports}
        missing_ids = [obs_id for obs_id in observation_ids if obs_id not in covered_ids]
        coverage_num = len(covered_ids & observation_id_set)
        coverage_den = max(len(observation_ids), 1)
        candidates.append({
            'entity_id': candidate_id,
            'support': _sample_items(supports, max_supports_per_candidate),
            'missing_ids': missing_ids,
            'coverage_num': coverage_num,
            'coverage_den': coverage_den,
            'score': coverage_num / coverage_den,
        })

    if len(observation_ids) > 1:
        multi_target_candidates = [
            candidate for candidate in candidates if candidate['coverage_num'] > 1
        ]
        if multi_target_candidates:
            candidates = multi_target_candidates
    return {
        'mode': 'common_cause',
        'graph_split': graph_split,
        'observation_ids': observation_ids,
        'candidates': _sample_items(candidates, top_k),
    }


def _incoming_source_edges(target_ids: list[int], sampler) -> list[dict]:
    target_id_set = set(int(target_id) for target_id in target_ids)
    edges = []
    for target_index, target_id in enumerate(target_ids):
        for source_id, object_id, relation_id in sampler.in_edges(int(target_id)):
            edge = {
                'subject_id': int(source_id),
                'relation_id': int(relation_id),
                'object_id': int(object_id),
                'observation_index': target_index,
            }
            if _is_self_loop(edge):
                continue
            if int(edge['subject_id']) in target_id_set:
                continue
            edges.append(edge)
    return _dedupe_edges(edges)


def build_alternative_evidence(
        target_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10):
    target_ids = observation_text_to_answer_ids(target_text, kg)
    sampler = _resolve_graph_sampler(kg, graph_split)
    incoming_edges = _incoming_source_edges(target_ids, sampler)
    source_ids = _sample_items(sorted({edge['subject_id'] for edge in incoming_edges}), top_k)
    outgoing_edges = []

    for source_id in source_ids:
        for subject_id, object_id, relation_id in sampler.out_edges(int(source_id)):
            edge = {
                'subject_id': int(subject_id),
                'relation_id': int(relation_id),
                'object_id': int(object_id),
            }
            if _is_self_loop(edge):
                continue
            outgoing_edges.append(edge)

    sampled_edges = _sample_items(_dedupe_edges(outgoing_edges), top_k)
    return {
        'mode': 'find_alternative',
        'graph_split': graph_split,
        'observation_ids': target_ids,
        'candidates': [_candidate_from_edge(edge) for edge in sampled_edges],
    }


def build_exclusion_evidence(
        target_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10):
    target_ids = observation_text_to_answer_ids(target_text, kg)
    target_id_set = set(int(target_id) for target_id in target_ids)
    sampler = _resolve_graph_sampler(kg, graph_split)
    incoming_edges = _incoming_source_edges(target_ids, sampler)
    relation_ids = sorted({edge['relation_id'] for edge in incoming_edges})

    contrast_edges = []
    for relation_id in relation_ids:
        edges_for_relation = []
        objects_by_source = defaultdict(set)
        for subject_id, object_id, edge_relation_id in sampler.graph.edges(keys=True):
            if int(edge_relation_id) != int(relation_id):
                continue
            subject_id = int(subject_id)
            object_id = int(object_id)
            objects_by_source[subject_id].add(object_id)
            edges_for_relation.append({
                'subject_id': subject_id,
                'relation_id': int(edge_relation_id),
                'object_id': object_id,
            })

        excluded_sources = {
            source_id
            for source_id, object_ids in objects_by_source.items()
            if object_ids & target_id_set
        }
        for edge in edges_for_relation:
            if _is_self_loop(edge):
                continue
            if int(edge['subject_id']) in excluded_sources:
                continue
            if int(edge['object_id']) in target_id_set:
                continue
            contrast_edges.append(edge)

    sampled_edges = _sample_items(_dedupe_edges(contrast_edges), top_k)
    return {
        'mode': 'find_exclusion',
        'graph_split': graph_split,
        'observation_ids': target_ids,
        'candidates': [_candidate_from_edge(edge) for edge in sampled_edges],
    }


def build_expand_evidence(
        target_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10,
        direction: str = 'backward',
        max_supports_per_candidate: int = 4):
    target_ids = observation_text_to_answer_ids(target_text, kg)
    sampler = _resolve_graph_sampler(kg, graph_split)
    direction = str(direction or 'backward').lower()
    support_by_candidate = defaultdict(list)

    for target_index, target_id in enumerate(target_ids):
        if direction == 'forward':
            edge_iter = (
                {
                    'subject_id': int(source_id),
                    'relation_id': int(relation_id),
                    'object_id': int(object_id),
                    'observation_index': target_index,
                }
                for source_id, object_id, relation_id in sampler.out_edges(int(target_id))
            )
            candidate_getter = lambda edge: edge['object_id']
        else:
            edge_iter = (
                {
                    'subject_id': int(source_id),
                    'relation_id': int(relation_id),
                    'object_id': int(object_id),
                    'observation_index': target_index,
                }
                for source_id, object_id, relation_id in sampler.in_edges(int(target_id))
            )
            candidate_getter = lambda edge: edge['subject_id']

        for edge in edge_iter:
            if _is_self_loop(edge):
                continue
            support_by_candidate[int(candidate_getter(edge))].append(edge)

    candidates = []
    target_id_set = set(target_ids)
    for candidate_id, supports in support_by_candidate.items():
        supports = _dedupe_edges(supports)
        if not supports:
            continue
        if direction == 'forward':
            covered_ids = {support['subject_id'] for support in supports}
        else:
            covered_ids = {support['object_id'] for support in supports}
        missing_ids = [target_id for target_id in target_ids if target_id not in covered_ids]
        coverage_num = len(covered_ids & target_id_set)
        coverage_den = max(len(target_ids), 1)
        candidates.append({
            'entity_id': candidate_id,
            'support': _sample_items(supports, max_supports_per_candidate),
            'missing_ids': missing_ids,
            'coverage_num': coverage_num,
            'coverage_den': coverage_den,
            'depths': [1],
            'score': coverage_num / coverage_den,
        })

    return {
        'mode': 'expand',
        'direction': direction,
        'graph_split': graph_split,
        'observation_ids': target_ids,
        'candidates': _sample_items(candidates, top_k),
    }


def render_evidence_package(evidence: dict, kg, prefix: str = 'RESULT') -> str:
    del prefix
    lines = []
    candidates = evidence.get('candidates', [])

    if not candidates:
        return tag_result_text('')

    edges = []
    for candidate in candidates:
        edges.extend(candidate.get('support', []))

    for edge in _dedupe_edges(edges):
        lines.append(_format_edge(edge, kg))

    return tag_result_text('\n'.join(lines))
