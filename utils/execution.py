from utils.logic_dsl import logic_text_to_query_wordlist


def compute_query_complexity(query_wordlist) -> int:
    return sum(1 for token in query_wordlist if token in {'p', 'i', 'u', 'n', 'e'})


def execute_logic_text(
        logic_text: str,
        kg,
        graph_samplers,
        split: str = 'train') -> dict:
    result = {
        'parse_success': 0.0,
        'execution_success': 0.0,
        'answers': [],
        'answer_count': 0,
        'query_complexity': 0,
        'error': '',
    }

    if not str(logic_text).strip():
        result['error'] = 'parse_error: empty logic text'
        return result

    try:
        query_wordlist = logic_text_to_query_wordlist(logic_text, kg)
    except Exception as exc:
        result['error'] = f'parse_error: {exc}'
        return result

    result['parse_success'] = 1.0
    result['query_complexity'] = compute_query_complexity(query_wordlist)

    try:
        sampler = graph_samplers[split]
        answers = sampler.search_answers_to_query(query_wordlist)
    except Exception as exc:
        result['error'] = f'execution_error: {exc}'
        return result

    result['execution_success'] = 1.0
    result['answers'] = list(answers)
    result['answer_count'] = len(result['answers'])
    return result


def compute_answer_set_scores(pred_answers, gold_answers) -> dict:
    pred_set = set(pred_answers)
    gold_set = set(gold_answers)
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    pred_count = len(pred_set)
    gold_count = len(gold_set)
    return {
        'answer_jaccard': intersection / union if union > 0 else 1.0,
        'answer_dice': (2.0 * intersection) / (pred_count + gold_count) if (pred_count + gold_count) > 0 else 1.0,
        'answer_overlap': intersection / max(min(pred_count, gold_count), 1),
        'answer_precision': intersection / max(pred_count, 1),
        'answer_recall': intersection / max(gold_count, 1),
    }


def score_logic_execution(
        logic_text: str,
        gold_answers,
        kg,
        graph_samplers,
        split: str = 'train',
        complexity_weight: float = 0.05) -> dict:
    execution = execute_logic_text(
        logic_text=logic_text,
        kg=kg,
        graph_samplers=graph_samplers,
        split=split,
    )
    scores = compute_answer_set_scores(execution['answers'], gold_answers)
    reward = (
        execution['parse_success']
        + execution['execution_success']
        + scores['answer_jaccard']
        + scores['answer_overlap']
        - complexity_weight * execution['query_complexity']
    )
    return {
        **execution,
        **scores,
        'logic_reward': float(reward),
    }
