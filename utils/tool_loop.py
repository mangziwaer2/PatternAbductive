from utils.kg_actions import execute_action_text, parse_action_text
from utils.logic_dsl import looks_like_dsl


def is_complete_action_text(action_text: str) -> bool:
    try:
        action = parse_action_text(action_text)
    except Exception:
        return False
    if not action.get('action_type'):
        return False
    if action.get('action_type') == 'CHECK_COVERAGE':
        return bool(action.get('candidates')) and bool(action.get('obs'))
    return bool(action.get('targets'))


def extract_action_text(generated_text: str) -> str | None:
    text = str(generated_text).strip()
    action_index = text.find('ACTION')
    if action_index < 0:
        return None
    text = text[action_index:]
    for delimiter in ['\nRESULT', '\nDSL', '\nTASK', '\nPATTERN']:
        delimiter_index = text.find(delimiter)
        if delimiter_index > 0:
            text = text[:delimiter_index]
    first_line = text.splitlines()[0].strip() if text.splitlines() else text.strip()
    if not first_line:
        return None
    return first_line if is_complete_action_text(first_line) else None


def extract_dsl_text(generated_text: str) -> str | None:
    text = str(generated_text).strip()
    dsl_index = text.find('DSL')
    if dsl_index >= 0:
        candidate = text[dsl_index + len('DSL'):].strip()
    else:
        candidate = text
    for delimiter in ['\nACTION', '\nRESULT', '\nTASK', '\nPATTERN']:
        delimiter_index = candidate.find(delimiter)
        if delimiter_index > 0:
            candidate = candidate[:delimiter_index]
    candidate = candidate.strip()
    if looks_like_dsl(candidate):
        return candidate
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
