from utils.condition import CONDITION_LABELS, extract_condition_metadata
from utils.evaluation import get_smatch_score
from utils.parsing import qry_wordlist_2_graph
from utils.textualization import (
    entity_text_to_id,
    observation_text_to_answer_ids,
    query_text_to_wordlist,
    relation_text_to_id,
)


LABEL_TO_TYPE = {label: condition_type for condition_type, label in CONDITION_LABELS.items()}


def parse_textual_condition(condition_text: str):
    tokens = str(condition_text or '').split()
    if not tokens:
        return {}
    if tokens[0] == 'COND':
        tokens = tokens[1:]

    parsed = {}
    index = 0
    while index < len(tokens):
        label = tokens[index]
        if label not in LABEL_TO_TYPE:
            index += 1
            continue
        index += 1
        value_tokens = []
        while index < len(tokens) and tokens[index] not in LABEL_TO_TYPE:
            value_tokens.append(tokens[index])
            index += 1
        parsed[LABEL_TO_TYPE[label]] = ' '.join(value_tokens).strip()
    return parsed


def compute_set_similarity(pred_items, gold_items):
    pred_set = set(pred_items)
    gold_set = set(gold_items)
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    pred_count = len(pred_set)
    gold_count = len(gold_set)
    return {
        'jaccard': intersection / union if union > 0 else 1.0,
        'dice': (2.0 * intersection) / (pred_count + gold_count) if (pred_count + gold_count) > 0 else 1.0,
        'overlap': intersection / max(min(pred_count, gold_count), 1),
    }


def compute_condition_satisfaction(pred_metadata: dict, condition_text: str, kg):
    conditions = parse_textual_condition(condition_text)
    if not conditions:
        return 1.0

    checks = []
    for condition_type, value in conditions.items():
        if condition_type == 'pattern':
            checks.append(pred_metadata['pattern_condition_text'] == value)
        elif condition_type == 'entity-number':
            checks.append(pred_metadata['entity_number'] == int(value))
        elif condition_type == 'relation-number':
            checks.append(pred_metadata['relation_number'] == int(value))
        elif condition_type == 'specific-entity':
            checks.append(entity_text_to_id(value, kg) in pred_metadata['unique_anchor_entities'])
        elif condition_type == 'specific-relation':
            checks.append(relation_text_to_id(value, kg) in pred_metadata['unique_relations'])
    return sum(checks) / len(checks) if checks else 1.0


def build_text_score_zero():
    return {
        'smatch': 0.0,
        'jaccard': 0.0,
        'dice': 0.0,
        'overlap': 0.0,
        'validity': 0.0,
        'condition': 0.0,
    }


def score_text_query_prediction(
        completion: str,
        target: str,
        source: str,
        condition_text: str,
        kg,
        graph_samplers,
        searching_split: str):
    score = build_text_score_zero()

    try:
        pred_wordlist = query_text_to_wordlist(completion, kg)
        target_wordlist = query_text_to_wordlist(target, kg)
    except Exception:
        return score

    pred_graph = qry_wordlist_2_graph(pred_wordlist)
    target_graph = qry_wordlist_2_graph(target_wordlist)
    if pred_graph is None or target_graph is None:
        return score

    try:
        target_answers = observation_text_to_answer_ids(source, kg)
        pred_answers = graph_samplers[searching_split].search_answers_to_query(pred_wordlist)
        pred_metadata = extract_condition_metadata(pred_wordlist)
        target_metadata = extract_condition_metadata(target_wordlist)
    except Exception:
        return score

    score.update(compute_set_similarity(pred_answers, target_answers))
    score['smatch'] = get_smatch_score(pred_graph, target_graph)
    score['validity'] = 1.0 if pred_metadata['pattern'] == target_metadata['pattern'] else 0.0
    score['condition'] = compute_condition_satisfaction(pred_metadata, condition_text, kg)
    return score


def score_text_query_batch(
        completions,
        targets,
        sources,
        condition_texts,
        kg,
        graph_samplers,
        searching_split: str):
    return [
        score_text_query_prediction(
            completion=completion,
            target=target,
            source=source,
            condition_text=condition_text,
            kg=kg,
            graph_samplers=graph_samplers,
            searching_split=searching_split,
        )
        for completion, target, source, condition_text in zip(completions, targets, sources, condition_texts)
    ]
