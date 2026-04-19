from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count


OPEN_TOKEN = '('
CLOSE_TOKEN = ')'
OPERATORS = {'p', 'i', 'u', 'n', 'e'}


class HypothesisParseError(ValueError):
    pass


@dataclass
class HypothesisNode:
    op: str
    value: str | None = None
    children: list['HypothesisNode'] = field(default_factory=list)


def _flip_relation_direction(relation: str) -> str:
    if relation.startswith('+'):
        return '-' + relation[1:]
    if relation.startswith('-'):
        return '+' + relation[1:]
    return relation


def parse_hypothesis_text(text: str) -> HypothesisNode:
    tokens = str(text).split()
    if not tokens:
        raise HypothesisParseError('Empty hypothesis text.')
    node, next_index = _parse_node(tokens, 0)
    if next_index != len(tokens):
        raise HypothesisParseError(
            f'Unexpected trailing tokens: {" ".join(tokens[next_index:])}'
        )
    return node


def _parse_node(tokens: list[str], index: int) -> tuple[HypothesisNode, int]:
    if index >= len(tokens) or tokens[index] != OPEN_TOKEN:
        raise HypothesisParseError(f'Expected "(" at token index {index}.')
    if index + 1 >= len(tokens):
        raise HypothesisParseError('Unexpected end of tokens after "(".')

    operator = tokens[index + 1]
    if operator not in OPERATORS:
        raise HypothesisParseError(f'Unsupported operator "{operator}".')

    if operator == 'e':
        if index + 3 >= len(tokens):
            raise HypothesisParseError('Incomplete entity expression.')
        entity = tokens[index + 2]
        if tokens[index + 3] != CLOSE_TOKEN:
            raise HypothesisParseError('Entity expression must end with ")".')
        return HypothesisNode(op='e', value=entity), index + 4

    if operator == 'p':
        if index + 2 >= len(tokens):
            raise HypothesisParseError('Incomplete projection expression.')
        relation = tokens[index + 2]
        child, next_index = _parse_node(tokens, index + 3)
        if next_index >= len(tokens) or tokens[next_index] != CLOSE_TOKEN:
            raise HypothesisParseError('Projection expression must end with ")".')
        return HypothesisNode(op='p', value=relation, children=[child]), next_index + 1

    if operator == 'n':
        child, next_index = _parse_node(tokens, index + 2)
        if next_index >= len(tokens) or tokens[next_index] != CLOSE_TOKEN:
            raise HypothesisParseError('Negation expression must end with ")".')
        return HypothesisNode(op='n', children=[child]), next_index + 1

    children = []
    next_index = index + 2
    while next_index < len(tokens) and tokens[next_index] != CLOSE_TOKEN:
        child, next_index = _parse_node(tokens, next_index)
        children.append(child)
    if next_index >= len(tokens) or tokens[next_index] != CLOSE_TOKEN:
        raise HypothesisParseError(f'Expression "{operator}" must end with ")".')
    if len(children) < 2:
        raise HypothesisParseError(f'Operator "{operator}" requires at least 2 children.')
    return HypothesisNode(op=operator, children=children), next_index + 1


def infer_pattern(node: HypothesisNode) -> str:
    if node.op == 'e':
        return '(e)'
    if node.op == 'p':
        return f'(p,{infer_pattern(node.children[0])})'
    if node.op == 'n':
        return f'(n,{infer_pattern(node.children[0])})'
    if node.op in {'i', 'u'}:
        child_patterns = ','.join(infer_pattern(child) for child in node.children)
        return f'({node.op},{child_patterns})'
    raise ValueError(f'Unsupported operator: {node.op}')


def collect_anchors(node: HypothesisNode) -> list[str]:
    if node.op == 'e':
        return [node.value] if node.value is not None else []
    anchors = []
    for child in node.children:
        anchors.extend(collect_anchors(child))
    return anchors


def collect_relations(node: HypothesisNode) -> list[str]:
    relations = []
    if node.op == 'p' and node.value is not None:
        relations.append(node.value)
    for child in node.children:
        relations.extend(collect_relations(child))
    return relations


def render_tree(node: HypothesisNode, indent: int = 0) -> list[str]:
    prefix = '  ' * indent
    if node.op == 'e':
        return [f'{prefix}ENTITY {node.value}']
    if node.op == 'p':
        lines = [f'{prefix}PROJECTION {node.value}']
    elif node.op == 'i':
        lines = [f'{prefix}INTERSECTION']
    elif node.op == 'u':
        lines = [f'{prefix}UNION']
    elif node.op == 'n':
        lines = [f'{prefix}NEGATION']
    else:
        lines = [f'{prefix}{node.op}']
    for child in node.children:
        lines.extend(render_tree(child, indent + 1))
    return lines


def build_logic_expression(node: HypothesisNode, variable: str = 'x') -> str:
    return _build_logic_expression(node, variable, count(1))


def _build_logic_expression(node: HypothesisNode, variable: str, counter) -> str:
    if node.op == 'e':
        return f'{variable} = `{node.value}`'

    if node.op == 'p':
        child = node.children[0]
        if child.op == 'e':
            if node.value and node.value.startswith('-'):
                return f'({variable} has relation `{_flip_relation_direction(node.value)}` to `{child.value}`)'
            if node.value and node.value.startswith('+'):
                return f'(`{child.value}` has relation `{node.value}` to {variable})'
            return f'({variable} is connected to `{child.value}` via `{node.value}`)'
        source_var = f'v{next(counter)}'
        child_expr = _build_logic_expression(child, source_var, counter)
        edge_expr = f'{source_var} has relation `{node.value}` to {variable}'
        return f'(exists {source_var}: {child_expr} AND {edge_expr})'

    if node.op == 'n':
        child_expr = _build_logic_expression(node.children[0], variable, counter)
        return f'NOT ({child_expr})'

    joiner = ' AND ' if node.op == 'i' else ' OR '
    child_exprs = [_build_logic_expression(child, variable, counter) for child in node.children]
    return '(' + joiner.join(child_exprs) + ')'


def _render_predicate(node: HypothesisNode) -> str:
    if node.op == 'e':
        return f'is exactly `{node.value}`'

    if node.op == 'p':
        child = node.children[0]
        if child.op == 'e':
            if node.value and node.value.startswith('-'):
                return f'has relation `{_flip_relation_direction(node.value)}` to `{child.value}`'
            if node.value and node.value.startswith('+'):
                return f'is reached from `{child.value}` via `{node.value}`'
            return f'is connected to `{child.value}` via `{node.value}`'
        return f'is reachable through `{node.value}` from something that { _render_predicate(child) }'

    if node.op == 'n':
        return f'not ({_render_predicate(node.children[0])})'

    joined = '; '.join(_render_predicate(child) for child in node.children)
    if node.op == 'i':
        return f'satisfy all of: {joined}'
    return f'satisfy at least one of: {joined}'


def build_readable_gloss(node: HypothesisNode) -> str:
    return 'entities that ' + _render_predicate(node)


def explain_hypothesis_text(text: str) -> dict:
    node = parse_hypothesis_text(text)
    anchors = collect_anchors(node)
    relations = collect_relations(node)
    return {
        'pattern': infer_pattern(node),
        'anchors': anchors,
        'relations': relations,
        'tree_lines': render_tree(node),
        'logic_expression': build_logic_expression(node),
        'gloss': build_readable_gloss(node),
    }
