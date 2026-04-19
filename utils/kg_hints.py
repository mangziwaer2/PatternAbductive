from __future__ import annotations

from collections import Counter

from utils.condition import CONDITION_LABELS
from utils.textualization import (
    KG_HINTS_PREFIX,
    KG_HINT_FACT_PREFIX,
    entity_id_to_text,
    normalize_symbol_name,
    observation_text_to_answer_ids,
)


def _parse_condition_preferences(condition_text: str) -> dict:
    preferences = {
        'specific_entity': None,
        'specific_relation': None,
    }
    if not condition_text:
        return preferences

    tokens = str(condition_text).split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == CONDITION_LABELS['specific-entity'] and i + 1 < len(tokens):
            preferences['specific_entity'] = tokens[i + 1]
            i += 2
            continue
        if token == CONDITION_LABELS['specific-relation'] and i + 1 < len(tokens):
            preferences['specific_relation'] = tokens[i + 1]
            i += 2
            continue
        i += 1
    return preferences


def _resolve_graph_split(kg, graph_split: str) -> str:
    if graph_split in kg.graph_samplers:
        return graph_split
    if 'train' in kg.graph_samplers:
        return 'train'
    return next(iter(kg.graph_samplers))


def _format_fact(subj_text: str, rel_text: str, obj_text: str) -> str:
    return f'{KG_HINT_FACT_PREFIX} {subj_text} {rel_text} {obj_text}'


def build_kg_hints_text(
        observation_text: str,
        kg,
        condition_text: str = '',
        graph_split: str = 'train',
        max_facts: int = 8) -> str:
    if kg is None or max_facts <= 0:
        return ''

    try:
        answer_ids = observation_text_to_answer_ids(observation_text, kg)
    except Exception:
        return ''
    if not answer_ids:
        return ''

    preferences = _parse_condition_preferences(condition_text)
    graph_split = _resolve_graph_split(kg, graph_split)
    graph = kg.graph_samplers[graph_split].graph

    out_edges_by_answer = {}
    shared_tail_counts = Counter()
    for answer_id in answer_ids:
        edges = list(graph.out_edges(answer_id, keys=True))
        out_edges_by_answer[answer_id] = edges
        for _, tail_id, _ in edges:
            shared_tail_counts[tail_id] += 1

    scored_facts = []
    seen = set()
    for answer_index, answer_id in enumerate(answer_ids):
        subj_text = entity_id_to_text(answer_id, kg)
        for _, tail_id, relation_id in out_edges_by_answer.get(answer_id, []):
            obj_text = entity_id_to_text(tail_id, kg)
            rel_text = normalize_symbol_name(kg.rel_id2name[int(relation_id)])
            fact_key = (subj_text, rel_text, obj_text)
            if fact_key in seen:
                continue
            seen.add(fact_key)

            score = 0
            if rel_text.startswith('+'):
                score += 5
            if len(answer_ids) > 1 and shared_tail_counts[tail_id] > 1:
                score += 50 + shared_tail_counts[tail_id]
            if preferences['specific_relation'] == rel_text:
                score += 100
            if preferences['specific_entity'] in {subj_text, obj_text}:
                score += 100
            score += max(0, 10 - answer_index)

            scored_facts.append((score, subj_text, rel_text, obj_text))

    if not scored_facts:
        return ''

    scored_facts.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    fact_texts = [
        _format_fact(subj_text, rel_text, obj_text)
        for _, subj_text, rel_text, obj_text in scored_facts[:max_facts]
    ]
    return f'{KG_HINTS_PREFIX} ' + ' '.join(fact_texts)


def build_batch_kg_hints_texts(
        observation_texts,
        kg,
        condition_texts=None,
        graph_split: str = 'train',
        max_facts: int = 8):
    if condition_texts is None:
        condition_texts = [''] * len(observation_texts)

    hints = []
    for observation_text, condition_text in zip(observation_texts, condition_texts):
        hints.append(build_kg_hints_text(
            observation_text=observation_text,
            kg=kg,
            condition_text=condition_text,
            graph_split=graph_split,
            max_facts=max_facts,
        ))
    return hints
