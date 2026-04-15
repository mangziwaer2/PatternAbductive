import torch
from transformers import LogitsProcessor

from utils.textualization import is_entity_text_token, is_relation_text_token


OPEN_TOKEN = '('
CLOSE_TOKEN = ')'
OPERATORS = ['p', 'i', 'u', 'n', 'e']

EXPECT_OPEN = 'expect_open'
EXPECT_OPERATOR = 'expect_operator'
EXPECT_ENTITY = 'expect_entity'
EXPECT_RELATION = 'expect_relation'
EXPECT_CLOSE = 'expect_close'
COMPLETE = 'complete'
INVALID = 'invalid'

PATTERN_LABEL = 'PATTERN'
ENTITY_NUM_LABEL = 'ENT_NUM'
REL_NUM_LABEL = 'REL_NUM'
SPEC_ENTITY_LABEL = 'SPEC_ENT'
SPEC_RELATION_LABEL = 'SPEC_REL'
STRUCTURE_TOKENS = [OPEN_TOKEN, CLOSE_TOKEN, *OPERATORS]


def classify_generated_prefix(tokens):
    stack = []
    state = EXPECT_OPEN

    for token in tokens:
        if state == EXPECT_OPEN:
            if token != OPEN_TOKEN:
                return INVALID, stack
            stack.append({'op': None, 'remaining_children': None})
            state = EXPECT_OPERATOR
            continue

        if state == EXPECT_OPERATOR:
            if token not in OPERATORS:
                return INVALID, stack
            node = stack[-1]
            node['op'] = token
            if token == 'e':
                state = EXPECT_ENTITY
            elif token == 'p':
                node['remaining_children'] = 1
                state = EXPECT_RELATION
            elif token == 'n':
                node['remaining_children'] = 1
                state = EXPECT_OPEN
            elif token in ['i', 'u']:
                node['remaining_children'] = 2
                state = EXPECT_OPEN
            continue

        if state == EXPECT_ENTITY:
            if not is_entity_text_token(token):
                return INVALID, stack
            state = EXPECT_CLOSE
            continue

        if state == EXPECT_RELATION:
            if not is_relation_text_token(token):
                return INVALID, stack
            state = EXPECT_OPEN
            continue

        if state == EXPECT_CLOSE:
            if token != CLOSE_TOKEN:
                return INVALID, stack
            stack.pop()
            if not stack:
                state = COMPLETE
            else:
                parent = stack[-1]
                if parent['remaining_children'] is None:
                    return INVALID, stack
                parent['remaining_children'] -= 1
                state = EXPECT_OPEN if parent['remaining_children'] > 0 else EXPECT_CLOSE
            continue

        if state == COMPLETE:
            return INVALID, stack

    return state, stack


def extract_structure_tokens(tokens):
    return [token for token in tokens if token in STRUCTURE_TOKENS]


def is_prefix_sequence(prefix_tokens, full_tokens):
    if len(prefix_tokens) > len(full_tokens):
        return False
    return prefix_tokens == full_tokens[:len(prefix_tokens)]


class TextConstraintState:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.open_id = tokenizer.convert_tokens_to_ids(OPEN_TOKEN)
        self.close_id = tokenizer.convert_tokens_to_ids(CLOSE_TOKEN)
        self.operator_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in OPERATORS}
        added_vocab = tokenizer.get_added_vocab()
        self.entity_ids = [token_id for token, token_id in added_vocab.items() if is_entity_text_token(token)]
        self.relation_ids = [token_id for token, token_id in added_vocab.items() if is_relation_text_token(token)]

        hypothesis_token_ids = [
            self.open_id,
            self.close_id,
            *self.operator_ids.values(),
            *self.entity_ids,
            *self.relation_ids,
        ]
        self.hypothesis_token_ids = sorted(set(token_id for token_id in hypothesis_token_ids if token_id is not None))
        self.all_hypothesis_tensor = None

    def parse_constraints(self, condition_text):
        constraints = {
            'entity_limit': None,
            'relation_limit': None,
            'required_entity_token': None,
            'required_relation_token': None,
            'pattern_tokens': None,
        }
        if not condition_text:
            return constraints

        tokens = condition_text.split()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == PATTERN_LABEL:
                j = i + 1
                pattern_tokens = []
                while j < len(tokens) and tokens[j] not in [PATTERN_LABEL, ENTITY_NUM_LABEL, REL_NUM_LABEL, SPEC_ENTITY_LABEL, SPEC_RELATION_LABEL]:
                    pattern_tokens.append(tokens[j])
                    j += 1
                constraints['pattern_tokens'] = pattern_tokens
                constraints['entity_limit'] = pattern_tokens.count('e')
                constraints['relation_limit'] = pattern_tokens.count('p')
                i = j
                continue
            if token == ENTITY_NUM_LABEL and i + 1 < len(tokens):
                constraints['entity_limit'] = int(tokens[i + 1])
                i += 2
                continue
            if token == REL_NUM_LABEL and i + 1 < len(tokens):
                constraints['relation_limit'] = int(tokens[i + 1])
                i += 2
                continue
            if token == SPEC_ENTITY_LABEL and i + 1 < len(tokens):
                constraints['required_entity_token'] = self.tokenizer.convert_tokens_to_ids(tokens[i + 1])
                i += 2
                continue
            if token == SPEC_RELATION_LABEL and i + 1 < len(tokens):
                constraints['required_relation_token'] = self.tokenizer.convert_tokens_to_ids(tokens[i + 1])
                i += 2
                continue
            i += 1
        return constraints

    def apply_pattern_prefix_constraint(self, allowed_token_ids, generated_ids, constraints):
        pattern_tokens = constraints.get('pattern_tokens')
        if not pattern_tokens:
            return allowed_token_ids

        current_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        current_structure = extract_structure_tokens(current_tokens)
        filtered_allowed = []
        for token_id in allowed_token_ids:
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            next_structure = current_structure + ([token] if token in STRUCTURE_TOKENS else [])
            if is_prefix_sequence(next_structure, pattern_tokens):
                filtered_allowed.append(token_id)
        return filtered_allowed

    def get_allowed_token_ids(self, generated_ids, constraints=None):
        if constraints is None:
            constraints = {
                'entity_limit': None,
                'relation_limit': None,
                'required_entity_token': None,
                'required_relation_token': None,
            }

        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        state, _ = classify_generated_prefix(tokens)
        entity_count = sum(is_entity_text_token(token) for token in tokens)
        relation_count = sum(is_relation_text_token(token) for token in tokens)
        required_entity_seen = constraints['required_entity_token'] is None or constraints['required_entity_token'] in generated_ids
        required_relation_seen = constraints['required_relation_token'] is None or constraints['required_relation_token'] in generated_ids

        if state == EXPECT_OPEN:
            allowed = [self.open_id]
            return self.apply_pattern_prefix_constraint(allowed, generated_ids, constraints)
        if state == EXPECT_OPERATOR:
            allowed = list(self.operator_ids.values())
            return self.apply_pattern_prefix_constraint(allowed, generated_ids, constraints)
        if state == EXPECT_ENTITY:
            if constraints['entity_limit'] is not None and entity_count >= constraints['entity_limit']:
                return []
            if (
                constraints['required_entity_token'] is not None
                and not required_entity_seen
                and constraints['entity_limit'] is not None
                and entity_count + 1 >= constraints['entity_limit']
            ):
                return [constraints['required_entity_token']]
            return self.entity_ids
        if state == EXPECT_RELATION:
            if constraints['relation_limit'] is not None and relation_count >= constraints['relation_limit']:
                return []
            if (
                constraints['required_relation_token'] is not None
                and not required_relation_seen
                and constraints['relation_limit'] is not None
                and relation_count + 1 >= constraints['relation_limit']
            ):
                return [constraints['required_relation_token']]
            return self.relation_ids
        if state == EXPECT_CLOSE:
            allowed = [self.close_id]
            return self.apply_pattern_prefix_constraint(allowed, generated_ids, constraints)
        if state == COMPLETE:
            pattern_tokens = constraints.get('pattern_tokens')
            if pattern_tokens is not None and extract_structure_tokens(tokens) != pattern_tokens:
                return []
            if constraints['entity_limit'] is not None and entity_count != constraints['entity_limit']:
                return []
            if constraints['relation_limit'] is not None and relation_count != constraints['relation_limit']:
                return []
            if not required_entity_seen or not required_relation_seen:
                return []
            return [self.tokenizer.eos_token_id]
        return self.hypothesis_token_ids


class TextConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_lengths, condition_texts=None):
        self.tokenizer = tokenizer
        self.prompt_lengths = prompt_lengths
        self.constraint_state = TextConstraintState(tokenizer)
        self.condition_constraints = []
        if condition_texts is None:
            condition_texts = [''] * len(prompt_lengths)
        for condition_text in condition_texts:
            self.condition_constraints.append(self.constraint_state.parse_constraints(condition_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_index in range(scores.shape[0]):
            prompt_length = self.prompt_lengths[batch_index]
            generated_ids = input_ids[batch_index, prompt_length:].tolist()
            constraints = self.condition_constraints[batch_index]
            allowed_token_ids = self.constraint_state.get_allowed_token_ids(generated_ids, constraints=constraints)
            if len(allowed_token_ids) == 0:
                continue
            mask = torch.full_like(scores[batch_index], float('-inf'))
            mask[allowed_token_ids] = 0.0
            scores[batch_index] = scores[batch_index] + mask
        return scores
