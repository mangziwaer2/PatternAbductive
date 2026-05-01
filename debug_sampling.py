import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Any

from utils.logic_dsl import query_wordlist_to_dsl
from utils.textualization import (
    format_surface_atom,
    observation_to_text,
    query_wordlist_to_text,
)


@dataclass
class SampleDebugResult:
    query: list[Any] | None = None
    tail_node: int | None = None
    answers: list[int] = field(default_factory=list)
    sampling_events: list[str] = field(default_factory=list)
    execution_events: list[str] = field(default_factory=list)


class PatternSamplingDebugger:
    """A small, debuggable copy of the core pattern -> query -> OBS flow.

    This intentionally does not write any dataset files. It mirrors the core
    recursion in utils.kgclass.GraphSampler, but keeps readable trace messages
    so the sampling process can be inspected step by step in a debugger.
    """

    def __init__(
            self,
            kg,
            split: str = 'train',
            max_set_preview: int = 12,
            branch_attempts: int = 16):
        self.kg = kg
        self.split = split
        self.sampler = kg.graph_samplers[split]
        self.max_set_preview = max_set_preview
        self.branch_attempts = branch_attempts
        self.sampling_events: list[str] = []
        self.execution_events: list[str] = []

    def sample_pair(
            self,
            pattern_str: str,
            max_answer_size: int = 8,
            max_tries: int = 200,
            tail_node: int | None = None,
            require_non_empty: bool = True,
            require_max_answer_size: bool = True) -> SampleDebugResult:
        if tail_node is not None:
            tries = [tail_node]
        else:
            tries = [random.choice(self.sampler.dense_nodes) for _ in range(max_tries)]

        last_result = SampleDebugResult()
        for attempt, candidate_tail in enumerate(tries, start=1):
            self.sampling_events = [f'attempt {attempt}: tail_node = {self.entity_text(candidate_tail)}']
            self.execution_events = []

            query, _, _ = self._sample_recursive(
                pattern=pattern_str,
                tail_node=candidate_tail,
                path='root',
                depth=0,
            )
            if query is None:
                last_result = SampleDebugResult(
                    query=None,
                    tail_node=candidate_tail,
                    sampling_events=list(self.sampling_events),
                )
                continue

            answers = self._execute_recursive(query=query, path='root', depth=0)
            last_result = SampleDebugResult(
                query=query,
                tail_node=candidate_tail,
                answers=answers,
                sampling_events=list(self.sampling_events),
                execution_events=list(self.execution_events),
            )

            if require_non_empty and len(answers) == 0:
                self.sampling_events.append('reject: answer set is empty')
                last_result.sampling_events = list(self.sampling_events)
                continue
            if require_max_answer_size and len(answers) > max_answer_size:
                self.sampling_events.append(
                    f'reject: answer size {len(answers)} > max_answer_size {max_answer_size}'
                )
                last_result.sampling_events = list(self.sampling_events)
                continue
            return last_result

        return last_result

    def _sample_recursive(self, pattern: str, tail_node: int, path: str, depth: int):
        operator, sub_pattern = self.sampler.extract_operator_subqueries(pattern)
        indent = '  ' * depth
        self.sampling_events.append(
            f'{indent}{path}: operator={operator}, target={self.entity_text(tail_node)}'
        )

        if operator == 'p':
            incoming_edges = list(self.sampler.in_edges(tail_node))
            if not incoming_edges:
                self.sampling_events.append(
                    f'{indent}{path}: fail, target has no incoming edges'
                )
                return None, None, None

            random.shuffle(incoming_edges)
            for head_node, _, relation in incoming_edges:
                self.sampling_events.append(
                    f'{indent}{path}: try edge '
                    f'{self.entity_text(head_node)} --{self.relation_text(relation)}--> '
                    f'{self.entity_text(tail_node)}'
                )
                self.sampling_events.append(
                    f'{indent}{path}: fill child with head={self.entity_text(head_node)}, '
                    f'query stores relation id {-relation}'
                )

                sub_query, _, prev_relation = self._sample_recursive(
                    pattern=sub_pattern[1],
                    tail_node=head_node,
                    path=f'{path}.child',
                    depth=depth + 1,
                )
                if sub_query is None:
                    self.sampling_events.append(
                        f'{indent}{path}: reject edge because child query failed'
                    )
                    continue
                if self.sampler.is_reverse_edge(prev_relation, relation):
                    self.sampling_events.append(
                        f'{indent}{path}: reject edge because relation {relation} '
                        f'immediately reverses previous relation {prev_relation}'
                    )
                    continue

                self.sampling_events.append(f'{indent}{path}: accept projection edge')
                return ['(', 'p', '(', -relation, ')', *sub_query, ')'], head_node, relation

            self.sampling_events.append(
                f'{indent}{path}: fail, no incoming edge produced a valid child query'
            )
            return None, None, None

        if operator == 'n':
            self.sampling_events.append(
                f'{indent}{path}: NOT does not choose a negative KG edge; '
                'it samples the inner query with the same target, then execution takes complement'
            )
            sub_query, head_node, relation = self._sample_recursive(
                pattern=sub_pattern[1],
                tail_node=tail_node,
                path=f'{path}.negated',
                depth=depth + 1,
            )
            if sub_query is None:
                return None, None, None
            return ['(', 'n', *sub_query, ')'], head_node, relation

        if operator == 'e':
            self.sampling_events.append(
                f'{indent}{path}: fill entity anchor e = {self.entity_text(tail_node)}'
            )
            return ['(', 'e', '(', tail_node, ')', ')'], None, None

        if operator == 'i':
            sub_queries_list = []
            from_node_list = []
            for index, child_pattern in enumerate(sub_pattern[1:], start=1):
                self.sampling_events.append(
                    f'{indent}{path}: intersection child {index} uses the same target '
                    f'{self.entity_text(tail_node)}'
                )
                sub_query, head_node, relation = None, None, None
                for attempt in range(1, self.branch_attempts + 1):
                    self.sampling_events.append(
                        f'{indent}{path}: child {index} attempt {attempt}/{self.branch_attempts}'
                    )
                    candidate_query, candidate_head, candidate_relation = self._sample_recursive(
                        pattern=child_pattern,
                        tail_node=tail_node,
                        path=f'{path}.and{index}',
                        depth=depth + 1,
                    )
                    if candidate_query is None:
                        continue
                    if candidate_query in sub_queries_list:
                        self.sampling_events.append(
                            f'{indent}{path}: reject child {index}, duplicated child query'
                        )
                        continue
                    if candidate_head in from_node_list:
                        self.sampling_events.append(
                            f'{indent}{path}: reject child {index}, duplicated source node '
                            f'{self.entity_text(candidate_head)}'
                        )
                        continue
                    sub_query, head_node, relation = candidate_query, candidate_head, candidate_relation
                    break

                if sub_query is None:
                    self.sampling_events.append(
                        f'{indent}{path}: fail, child {index} cannot produce a non-duplicate query'
                    )
                    return None, None, None
                sub_queries_list.append(sub_query)
                from_node_list.append(head_node)

            result = ['(', 'i']
            for sub_query in sub_queries_list:
                result.extend(sub_query)
            result.append(')')
            return result, tail_node, None

        if operator == 'u':
            sub_queries_list = []
            random_subquery_index = random.randint(1, len(sub_pattern) - 1)
            self.sampling_events.append(
                f'{indent}{path}: union child {random_subquery_index} is forced to contain '
                f'{self.entity_text(tail_node)}; other children use random targets'
            )

            for index in range(1, len(sub_pattern)):
                child_tail = tail_node
                if index != random_subquery_index:
                    child_tail = random.choice(list(self.sampler.graph.nodes()))

                sub_query, _, relation = self._sample_recursive(
                    pattern=sub_pattern[index],
                    tail_node=child_tail,
                    path=f'{path}.or{index}',
                    depth=depth + 1,
                )
                if sub_query is None:
                    return None, None, None
                sub_queries_list.append(sub_query)

            result = ['(', 'u']
            for sub_query in sub_queries_list:
                result.extend(sub_query)
            result.append(')')
            return result, tail_node, None

        raise ValueError(f'Unsupported pattern operator: {operator}')

    def _execute_recursive(self, query: list[Any], path: str, depth: int) -> list[int]:
        operator, sub_queries = self.sampler.extract_operator_subqueries(query)
        indent = '  ' * depth

        if operator == 'e':
            answer = [sub_queries[1][1]]
            self.execution_events.append(
                f'{indent}{path}: e -> {self.answer_set_text(answer)}'
            )
            return answer

        if operator == 'p':
            sub_answers = self._execute_recursive(
                query=sub_queries[2],
                path=f'{path}.child',
                depth=depth + 1,
            )
            relation_name = -sub_queries[1][1]
            all_answers = []
            for _, v, k in self.sampler.out_edges(sub_answers):
                if k == relation_name:
                    all_answers.append(v)
            answers = sorted(set(all_answers))
            self.execution_events.append(
                f'{indent}{path}: p relation={self.relation_text(relation_name)} '
                f'from {self.answer_set_text(sub_answers)} -> {self.answer_set_text(answers)}'
            )
            return answers

        if operator == 'n':
            inner_answers = self._execute_recursive(
                query=sub_queries[1],
                path=f'{path}.negated',
                depth=depth + 1,
            )
            all_nodes = set(self.sampler.graph.nodes)
            answers = sorted(all_nodes - set(inner_answers))
            self.execution_events.append(
                f'{indent}{path}: n complement ALL_NODES({len(all_nodes)}) - '
                f'{self.answer_set_text(inner_answers)} -> size={len(answers)}, '
                f'preview={self.answer_set_text(answers)}'
            )
            return answers

        if operator == 'i':
            child_answer_sets = []
            for index in range(1, len(sub_queries)):
                child_answers = self._execute_recursive(
                    query=sub_queries[index],
                    path=f'{path}.and{index}',
                    depth=depth + 1,
                )
                child_answer_sets.append(set(child_answers))

            merged = set(child_answer_sets[0]) if child_answer_sets else set()
            for child_answers in child_answer_sets[1:]:
                merged &= child_answers
            answers = sorted(merged)
            self.execution_events.append(
                f'{indent}{path}: i intersection -> {self.answer_set_text(answers)}'
            )
            return answers

        if operator == 'u':
            merged = set()
            for index in range(1, len(sub_queries)):
                child_answers = self._execute_recursive(
                    query=sub_queries[index],
                    path=f'{path}.or{index}',
                    depth=depth + 1,
                )
                merged |= set(child_answers)
            answers = sorted(merged)
            self.execution_events.append(
                f'{indent}{path}: u union -> {self.answer_set_text(answers)}'
            )
            return answers

        raise ValueError(f'Unsupported query operator: {operator}')

    def entity_text(self, entity_id: int) -> str:
        name = self.kg.ent_id2name.get(int(entity_id), str(entity_id))
        return f'{format_surface_atom(name)}#{int(entity_id)}'

    def relation_text(self, relation_id: int) -> str:
        name = self.kg.rel_id2name.get(int(relation_id), str(relation_id))
        return f'{format_surface_atom(name)}#{int(relation_id)}'

    def answer_set_text(self, answers: list[int] | set[int]) -> str:
        answer_list = sorted(int(answer) for answer in answers)
        shown = answer_list[:self.max_set_preview]
        body = ', '.join(self.entity_text(answer) for answer in shown)
        if len(answer_list) > len(shown):
            body += f', ... +{len(answer_list) - len(shown)}'
        return '{' + body + f'}} size={len(answer_list)}'


def resolve_tail_node(kg, tail_id: int | None, tail_name: str | None) -> int | None:
    if tail_id is not None:
        return int(tail_id)
    if not tail_name:
        return None
    normalized = ' '.join(tail_name.replace('_', ' ').split()).lower()
    matches = [
        entity_id
        for entity_id, entity_name in kg.ent_id2name.items()
        if ' '.join(str(entity_name).replace('_', ' ').split()).lower() == normalized
    ]
    if not matches:
        raise ValueError(f'Cannot find entity by name: {tail_name}')
    if len(matches) > 1:
        print(f'# Warning: multiple entity ids match "{tail_name}", using {matches[0]}')
    return int(matches[0])


def main():
    parser = argparse.ArgumentParser(description='Debug one pattern-based OBS/hypothesis sampling step.')
    parser.add_argument('--pattern', default='(i,(n,(p,(e))),(p,(e)))')
    parser.add_argument('--dataname', default='DBpedia50')
    parser.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-answer-size', type=int, default=8)
    parser.add_argument('--max-tries', type=int, default=200)
    parser.add_argument('--tail-id', type=int, default=None)
    parser.add_argument('--tail-name', default=None)
    parser.add_argument('--allow-empty', action='store_true')
    parser.add_argument('--allow-large', action='store_true')
    parser.add_argument('--preview', type=int, default=12)
    parser.add_argument('--branch-attempts', type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    from utils.load import load_kg

    kg = load_kg(args.dataname)
    tail_node = resolve_tail_node(kg, args.tail_id, args.tail_name)

    debugger = PatternSamplingDebugger(
        kg=kg,
        split=args.split,
        max_set_preview=args.preview,
        branch_attempts=args.branch_attempts,
    )
    result = debugger.sample_pair(
        pattern_str=args.pattern,
        max_answer_size=args.max_answer_size,
        max_tries=args.max_tries,
        tail_node=tail_node,
        require_non_empty=not args.allow_empty,
        require_max_answer_size=not args.allow_large,
    )

    print('\n# Pattern')
    print(args.pattern)

    print('\n# Sampling Trace')
    for event in result.sampling_events:
        print(event)

    if result.query is None:
        print('\n# No query sampled. Increase --max-tries or choose another --tail-id.')
        return

    print('\n# Concrete Query Wordlist')
    print(json.dumps(result.query, ensure_ascii=False))

    print('\n# Concrete Query Surface')
    print(query_wordlist_to_text(result.query, kg))

    print('\n# Concrete DSL')
    print(query_wordlist_to_dsl(result.query, kg))

    print('\n# Execution Trace')
    for event in result.execution_events:
        print(event)

    print('\n# Final OBS / Hypothesis Pair')
    print(observation_to_text(result.answers, kg))
    print('<DSL>')
    print(query_wordlist_to_dsl(result.query, kg))
    print('</DSL>')

    print('\n# Summary')
    print(f'split={args.split}')
    print(f'tail_node={debugger.entity_text(result.tail_node)}')
    print(f'answer_size={len(result.answers)}')


if __name__ == '__main__':
    main()
