import re

from utils.action_supervision import (
    ACTION_END_TAG,
    ACTION_START_TAG,
    VALID_ACTION_TYPES,
    normalize_action_type,
    strip_action_tags,
)
from utils.kg_actions import parse_action_text
from utils.logic_dsl import DSL_END_TAG, DSL_START_TAG
from utils.textualization import entity_text_to_id, is_relation_text_token, tokenize_surface_text


STRICT_ACTION_TYPES = {'FIND_COMMON', 'FIND_ALTERNATIVE', 'FIND_EXCLUSION', 'EXPAND'}
VALID_DIRECTIONS = {'forward', 'backward'}


def _safe_tokenize(text: str) -> list[str]:
    try:
        return tokenize_surface_text(text)
    except Exception:
        return []


def _is_entity_token(token: str) -> bool:
    return isinstance(token, str) and token.startswith('[') and token.endswith(']') and not is_relation_text_token(token)


def _is_relation_token(token: str) -> bool:
    return isinstance(token, str) and token.startswith('[') and token.endswith(']') and is_relation_text_token(token)


def _bracket_tokens(text: str) -> list[str]:
    return re.findall(r'\[[^\[\]\n]+\]', str(text or ''))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _extract_observation_tokens(observation_text: str) -> set[str]:
    return {
        token
        for token in _bracket_tokens(observation_text)
        if _is_entity_token(token)
    }


def strict_parse_action(action_text: str) -> dict:
    """Return strict grammar diagnostics without using the permissive action parser."""
    result = {
        'strict_action_parse': 0.0,
        'strict_action_error': '',
        'action_type': '',
        'targets': [],
    }

    body = strip_action_tags(action_text)
    tokens = _safe_tokenize(body)
    if len(tokens) < 6:
        result['strict_action_error'] = 'too_few_tokens'
        return result
    if tokens[0] != 'ACTION':
        result['strict_action_error'] = 'missing_ACTION_prefix'
        return result

    action_type = normalize_action_type(tokens[1])
    result['action_type'] = action_type
    if action_type not in STRICT_ACTION_TYPES:
        result['strict_action_error'] = 'unsupported_action_type'
        return result

    try:
        targets_index = tokens.index('TARGETS')
    except ValueError:
        result['strict_action_error'] = 'missing_TARGETS'
        return result
    if targets_index != 2:
        result['strict_action_error'] = 'unexpected_tokens_before_TARGETS'
        return result

    if action_type == 'EXPAND':
        try:
            direction_index = tokens.index('DIRECTION')
            top_k_index = tokens.index('TOP_K')
        except ValueError:
            result['strict_action_error'] = 'missing_DIRECTION_or_TOP_K'
            return result
        if not (targets_index < direction_index < top_k_index):
            result['strict_action_error'] = 'invalid_EXPAND_field_order'
            return result
        target_tokens = tokens[targets_index + 1:direction_index]
        if top_k_index != direction_index + 2:
            result['strict_action_error'] = 'invalid_DIRECTION_payload'
            return result
        if str(tokens[direction_index + 1]).lower() not in VALID_DIRECTIONS:
            result['strict_action_error'] = 'invalid_direction'
            return result
    else:
        try:
            top_k_index = tokens.index('TOP_K')
        except ValueError:
            result['strict_action_error'] = 'missing_TOP_K'
            return result
        if top_k_index <= targets_index + 1:
            result['strict_action_error'] = 'missing_targets'
            return result
        if any(token in {'DIRECTION', 'TARGETS'} for token in tokens[targets_index + 1:top_k_index]):
            result['strict_action_error'] = 'unexpected_field_token'
            return result
        target_tokens = tokens[targets_index + 1:top_k_index]

    if top_k_index != len(tokens) - 2:
        result['strict_action_error'] = 'trailing_tokens_after_TOP_K'
        return result
    try:
        top_k = int(tokens[top_k_index + 1])
    except Exception:
        result['strict_action_error'] = 'invalid_TOP_K'
        return result
    if top_k <= 0:
        result['strict_action_error'] = 'non_positive_TOP_K'
        return result

    if not target_tokens or not all(_is_entity_token(token) for token in target_tokens):
        result['strict_action_error'] = 'invalid_target_tokens'
        return result

    result['targets'] = target_tokens
    result['strict_action_parse'] = 1.0
    return result


def extract_action_type_sequence(history: list[str]) -> list[str]:
    sequence = []
    for index in range(0, len(history), 2):
        try:
            action = parse_action_text(history[index])
        except Exception:
            continue
        action_type = normalize_action_type(action.get('action_type', ''))
        if action_type:
            sequence.append(action_type)
    return sequence


def extract_trace_entities_and_relations(history: list[str], dsl_text: str, observation_text: str) -> dict:
    result_text = '\n'.join(str(history[index]) for index in range(1, len(history), 2))
    evidence_tokens = set(_bracket_tokens(result_text))
    dsl_tokens = set(_bracket_tokens(dsl_text))
    observation_tokens = _extract_observation_tokens(observation_text)
    return {
        'evidence_tokens': evidence_tokens,
        'dsl_tokens': dsl_tokens,
        'observation_tokens': observation_tokens,
        'evidence_entity_tokens': {token for token in evidence_tokens if _is_entity_token(token)},
        'evidence_relation_tokens': {token for token in evidence_tokens if _is_relation_token(token)},
        'dsl_entity_tokens': {token for token in dsl_tokens if _is_entity_token(token)},
        'dsl_relation_tokens': {token for token in dsl_tokens if _is_relation_token(token)},
    }


def score_trace_dsl_consistency(history: list[str], dsl_text: str, observation_text: str) -> dict:
    del observation_text
    action_types = extract_action_type_sequence(history)
    has_not = 'NOT(' in str(dsl_text)
    has_or = 'OR(' in str(dsl_text)
    has_expand = 'EXPAND' in action_types
    has_exclusion = 'FIND_EXCLUSION' in action_types
    has_alternative = 'FIND_ALTERNATIVE' in action_types

    components = []
    if has_not or has_exclusion:
        components.append(1.0 if has_not and has_exclusion else (0.5 if has_exclusion else 0.0))
    if has_or or has_alternative:
        components.append(1.0 if has_or and has_alternative else (0.5 if has_alternative else 0.0))

    token_sets = extract_trace_entities_and_relations(history, dsl_text, '')
    expand_overlap = 0.0
    if has_expand:
        expand_overlap = 1.0 if (token_sets['dsl_tokens'] & token_sets['evidence_tokens']) else 0.0
        components.append(expand_overlap)

    # Neutral score for simple common traces without NOT/OR/EXPAND requirements.
    consistency = sum(components) / len(components) if components else 0.5
    return {
        'trace_dsl_consistency': float(consistency),
        'not_consistency': float(1.0 if has_not and has_exclusion else 0.0 if has_not else 0.5 if has_exclusion else 0.5),
        'or_consistency': float(1.0 if has_or and has_alternative else 0.0 if has_or else 0.5 if has_alternative else 0.5),
        'expand_consistency': float(expand_overlap if has_expand else 0.5),
        'action_type_sequence': action_types,
    }


def score_evidence_usage(history: list[str], dsl_text: str, observation_text: str, kg=None) -> dict:
    token_sets = extract_trace_entities_and_relations(history, dsl_text, observation_text)
    dsl_tokens = token_sets['dsl_tokens']
    evidence_tokens = token_sets['evidence_tokens']
    observation_tokens = token_sets['observation_tokens']

    if dsl_tokens:
        evidence_usage = len(dsl_tokens & evidence_tokens) / len(dsl_tokens)
        hallucinated = dsl_tokens - evidence_tokens - observation_tokens
        hallucinated_rate = len(hallucinated) / len(dsl_tokens)
    else:
        evidence_usage = 0.0
        hallucinated = set()
        hallucinated_rate = 0.0

    unknown_entities = []
    if kg is not None:
        for token in sorted(token_sets['dsl_entity_tokens']):
            try:
                entity_text_to_id(token, kg)
            except Exception:
                unknown_entities.append(token)

    return {
        'evidence_usage': float(evidence_usage),
        'hallucinated_dsl_token_rate': float(hallucinated_rate),
        'hallucinated_dsl_tokens': sorted(hallucinated),
        'unknown_dsl_entity_count': len(unknown_entities),
    }


def score_step_control(history: list[str], dsl_text: str) -> dict:
    action_texts = [str(history[index]).strip() for index in range(0, len(history), 2)]
    normalized_actions = [strip_action_tags(action_text) for action_text in action_texts if action_text]
    repeat_count = len(normalized_actions) - len(set(normalized_actions))
    num_actions = len(normalized_actions)
    has_action = 1.0 if num_actions > 0 else 0.0
    has_final_dsl = 1.0 if str(dsl_text).strip() else 0.0
    step_control = (
        0.2 * has_action
        + 0.2 * has_final_dsl
        - 0.05 * max(num_actions - 2, 0)
        - 0.2 * max(repeat_count, 0)
    )
    return {
        'step_control': float(max(min(step_control, 1.0), -1.0)),
        'repeat_action_count': int(repeat_count),
    }


def score_action_strictness(history: list[str], observation_text: str, kg=None) -> dict:
    del observation_text
    action_texts = [str(history[index]).strip() for index in range(0, len(history), 2)]
    if not action_texts:
        return {
            'strict_action_parse_rate': 0.0,
            'malformed_action_rate': 0.0,
            'unknown_action_entity_count': 0,
            'strict_action_errors': [],
        }

    strict_scores = []
    errors = []
    unknown_entities = []
    for action_text in action_texts:
        strict = strict_parse_action(action_text)
        strict_scores.append(float(strict['strict_action_parse']))
        if strict['strict_action_error']:
            errors.append(strict['strict_action_error'])
        if kg is not None:
            for target in strict.get('targets') or []:
                try:
                    entity_text_to_id(target, kg)
                except Exception:
                    unknown_entities.append(target)

    strict_rate = sum(strict_scores) / len(strict_scores)
    return {
        'strict_action_parse_rate': float(strict_rate),
        'malformed_action_rate': float(1.0 - strict_rate),
        'unknown_action_entity_count': len(unknown_entities),
        'strict_action_errors': _dedupe(errors),
    }


def score_bad_generation(raw_generations: list[dict], history: list[str], dsl_text: str, kg=None) -> dict:
    del kg
    tag_role_mismatch = 0
    repeated_close_tag_count = 0
    malformed_action_text_count = 0

    for generation in raw_generations or []:
        generated = str(generation.get('generated', '') or '')
        repeated_close_tag_count += max(generated.count(ACTION_END_TAG) - 1, 0)
        repeated_close_tag_count += max(generated.count(DSL_END_TAG) - 1, 0)

        action_start = generated.find(ACTION_START_TAG)
        action_end = generated.find(ACTION_END_TAG, action_start + len(ACTION_START_TAG))
        if action_start >= 0 and action_end >= 0:
            body = generated[action_start + len(ACTION_START_TAG):action_end].strip()
            if body and not body.startswith('ACTION '):
                tag_role_mismatch += 1
            if body.startswith(('AND(', 'OR(', 'NOT(', 'PROJ(', 'ENT(')):
                tag_role_mismatch += 1

        dsl_start = generated.find(DSL_START_TAG)
        dsl_end = generated.find(DSL_END_TAG, dsl_start + len(DSL_START_TAG))
        if dsl_start >= 0 and dsl_end >= 0:
            body = generated[dsl_start + len(DSL_START_TAG):dsl_end].strip()
            if body.startswith('ACTION '):
                tag_role_mismatch += 1

    for action_text in [str(history[index]).strip() for index in range(0, len(history), 2)]:
        if strict_parse_action(action_text)['strict_action_parse'] <= 0:
            malformed_action_text_count += 1

    bad_generation_penalty = (
        1.0 * malformed_action_text_count
        + 0.8 * tag_role_mismatch
        + 0.3 * repeated_close_tag_count
    )
    if not str(dsl_text).strip():
        bad_generation_penalty += 0.8

    return {
        'tag_role_mismatch': int(tag_role_mismatch),
        'repeated_close_tag_count': int(repeated_close_tag_count),
        'malformed_action_text_count': int(malformed_action_text_count),
        'bad_generation_penalty': float(bad_generation_penalty),
    }
