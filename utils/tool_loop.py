from utils.kg_actions import execute_action_text, parse_action_text
from utils.logic_dsl import (
    DSL_END_TAG,
    DSL_START_TAG,
    extract_dsl_text as normalize_dsl_text,
    looks_like_dsl,
)
from utils.action_supervision import (
    ACTION_END_TAG,
    ACTION_START_TAG,
    action_has_explicit_boundary,
)


def is_complete_action_text(action_text: str) -> bool:
    if not action_has_explicit_boundary(action_text):
        return False
    try:
        action = parse_action_text(action_text)
    except Exception:
        return False
    return bool(action.get('action_type')) and bool(action.get('targets'))


def extract_action_text(generated_text: str) -> str | None:
    text = str(generated_text).strip()
    tag_index = text.find(ACTION_START_TAG)
    if tag_index < 0:
        return None
    end_index = text.find(ACTION_END_TAG, tag_index + len(ACTION_START_TAG))
    if end_index < 0:
        return None
    candidate = text[tag_index:end_index + len(ACTION_END_TAG)].strip()
    return candidate if is_complete_action_text(candidate) else None


def extract_dsl_text(generated_text: str) -> str | None:
    text = str(generated_text).strip()
    tag_index = text.find(DSL_START_TAG)
    if tag_index >= 0:
        end_index = text.find(DSL_END_TAG, tag_index + len(DSL_START_TAG))
        if end_index < 0:
            return None
        candidate = text[tag_index:end_index + len(DSL_END_TAG)].strip()
        dsl_text = normalize_dsl_text(candidate)
        return dsl_text if looks_like_dsl(dsl_text) else None

    return None


def build_tool_augmented_context(
        observation_text: str,
        action_text: str,
        result_text: str) -> str:
    return '\n'.join([
        str(observation_text).strip(),
        str(action_text).strip(),
        str(result_text).strip(),
    ]).strip()


def run_action_tool_call(
        observation_text: str,
        action_text: str,
        kg,
        graph_split: str = 'train') -> dict:
    if not is_complete_action_text(action_text):
        raise ValueError(f'Incomplete action text: {action_text}')

    action = parse_action_text(action_text)
    result_text = execute_action_text(action_text, kg=kg, graph_split=graph_split)
    return {
        'action': action,
        'action_text': action_text,
        'result_text': result_text,
        'next_context': build_tool_augmented_context(
            observation_text=observation_text,
            action_text=action_text,
            result_text=result_text,
        ),
    }
