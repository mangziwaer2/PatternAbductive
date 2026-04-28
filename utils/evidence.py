from collections import defaultdict

from utils.textualization import (
    entity_id_to_text,
    format_surface_atom,
    normalize_symbol_name,
    observation_text_to_answer_ids,
)


EVIDENCE_CONTROL_TOKENS = [
    'RESULT',
    'EVIDENCE',
    'CANDIDATE',
    'SUPPORT',
    'PATH',
    'MISSING',
    'MISSING_COUNT',
    'COVERAGE',
    'DEPTHS',
]


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


def _format_path(path_edges: list[dict], kg) -> str:
    if not path_edges:
        return ''
    parts = [_entity_text(path_edges[0]['subject_id'], kg)]
    for edge in path_edges:
        parts.append(f'--{_relation_text(edge["relation_id"], kg)}-->')
        parts.append(_entity_text(edge['object_id'], kg))
    return ' '.join(parts)


def _find_reverse_paths_to_observation(
        sampler,
        observation_id: int,
        max_depth: int,
        allowed_depths: set[int] | None = None,
        max_paths: int = 256):
    paths = []
    frontier = [(int(observation_id), [], {int(observation_id)})]

    for _ in range(max_depth):
        next_frontier = []
        for current_id, current_path, visited in frontier:
            for source_id, _, relation_id in sampler.in_edges(current_id):
                source_id = int(source_id)
                if source_id in visited:
                    continue
                edge = {
                    'subject_id': source_id,
                    'relation_id': int(relation_id),
                    'object_id': int(current_id),
                }
                new_path = [edge, *current_path]
                depth = len(new_path)
                if allowed_depths is None or depth in allowed_depths:
                    paths.append(new_path)
                    if len(paths) >= max_paths:
                        return paths
                if depth < max_depth:
                    next_frontier.append((source_id, new_path, visited | {source_id}))
        frontier = next_frontier
        if not frontier:
            break

    return paths


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
            support_by_candidate[int(source_id)].append({
                'subject_id': int(source_id),
                'relation_id': int(relation_id),
                'object_id': int(obs_id),
                'observation_index': obs_index,
            })

    candidates = []
    observation_id_set = set(observation_ids)
    for candidate_id, supports in support_by_candidate.items():
        covered_ids = {support['object_id'] for support in supports}
        missing_ids = [obs_id for obs_id in observation_ids if obs_id not in covered_ids]
        coverage_num = len(covered_ids & observation_id_set)
        coverage_den = max(len(observation_ids), 1)
        candidates.append({
            'entity_id': candidate_id,
            'support': supports[:max_supports_per_candidate],
            'missing_ids': missing_ids,
            'coverage_num': coverage_num,
            'coverage_den': coverage_den,
            'score': coverage_num / coverage_den,
        })

    candidates.sort(key=lambda item: (-item['coverage_num'], len(item['missing_ids']), item['entity_id']))
    return {
        'mode': 'common_cause',
        'graph_split': graph_split,
        'observation_ids': observation_ids,
        'candidates': candidates[:top_k],
    }


def build_path_evidence(
        observation_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10,
        max_hops: int = 2,
        branch_hops: list[int] | None = None,
        max_paths_per_observation: int = 256,
        max_paths_per_candidate: int = 4):
    observation_ids = observation_text_to_answer_ids(observation_text, kg)
    sampler = _resolve_graph_sampler(kg, graph_split)
    allowed_depths = {int(depth) for depth in (branch_hops or []) if int(depth) > 0}
    if not allowed_depths:
        allowed_depths = set(range(1, max(int(max_hops), 1) + 1))
    max_depth = max(allowed_depths) if allowed_depths else max(int(max_hops), 1)

    paths_by_candidate = defaultdict(list)
    for obs_index, obs_id in enumerate(observation_ids):
        paths = _find_reverse_paths_to_observation(
            sampler=sampler,
            observation_id=obs_id,
            max_depth=max_depth,
            allowed_depths=allowed_depths,
            max_paths=max_paths_per_observation,
        )
        for path_edges in paths:
            if not path_edges:
                continue
            candidate_id = int(path_edges[0]['subject_id'])
            paths_by_candidate[candidate_id].append({
                'observation_index': obs_index,
                'object_id': int(obs_id),
                'depth': len(path_edges),
                'edges': path_edges,
            })

    candidates = []
    observation_id_set = set(observation_ids)
    for candidate_id, paths in paths_by_candidate.items():
        covered_ids = {path['object_id'] for path in paths}
        missing_ids = [obs_id for obs_id in observation_ids if obs_id not in covered_ids]
        coverage_num = len(covered_ids & observation_id_set)
        coverage_den = max(len(observation_ids), 1)
        depths = sorted({path['depth'] for path in paths})
        candidates.append({
            'entity_id': candidate_id,
            'paths': sorted(paths, key=lambda item: (item['depth'], item['object_id']))[:max_paths_per_candidate],
            'missing_ids': missing_ids,
            'coverage_num': coverage_num,
            'coverage_den': coverage_den,
            'depths': depths,
            'score': coverage_num / coverage_den,
        })

    candidates.sort(
        key=lambda item: (
            -item['coverage_num'],
            len(item['missing_ids']),
            min(item.get('depths') or [999]),
            item['entity_id'],
        )
    )
    return {
        'mode': 'path',
        'graph_split': graph_split,
        'observation_ids': observation_ids,
        'branch_hops': sorted(allowed_depths),
        'candidates': candidates[:top_k],
    }


def render_evidence_package(evidence: dict, kg, prefix: str = 'RESULT') -> str:
    lines = [prefix]
    candidates = evidence.get('candidates', [])
    observation_ids = evidence.get('observation_ids', [])

    if not candidates:
        lines.append('NO_CANDIDATES')
        return '\n'.join(lines)

    for candidate in candidates:
        lines.append(f'CANDIDATE {_entity_text(candidate["entity_id"], kg)}')
        if candidate.get('depths'):
            lines.append('DEPTHS ' + ' '.join(str(depth) for depth in candidate['depths']))
        for support in candidate.get('support', []):
            lines.append(f'SUPPORT {_format_edge(support, kg)}')
        for path in candidate.get('paths', []):
            lines.append(f'PATH depth={path["depth"]} {_format_path(path["edges"], kg)}')
        missing_ids = candidate.get('missing_ids', [])
        if missing_ids:
            lines.append(f'MISSING_COUNT {len(missing_ids)}')
        coverage_den = candidate.get('coverage_den', max(len(observation_ids), 1))
        lines.append(f'COVERAGE {candidate.get("coverage_num", 0)}/{coverage_den}')
    return '\n'.join(lines)


def build_common_cause_result_text(
        observation_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10,
        max_hops: int = 2) -> str:
    evidence = build_common_cause_evidence(
        observation_text=observation_text,
        kg=kg,
        graph_split=graph_split,
        top_k=top_k,
        max_hops=max_hops,
    )
    return render_evidence_package(evidence, kg, prefix='RESULT')


def build_path_result_text(
        observation_text: str,
        kg,
        graph_split: str = 'train',
        top_k: int = 10,
        max_hops: int = 2,
        branch_hops: list[int] | None = None) -> str:
    evidence = build_path_evidence(
        observation_text=observation_text,
        kg=kg,
        graph_split=graph_split,
        top_k=top_k,
        max_hops=max_hops,
        branch_hops=branch_hops,
    )
    return render_evidence_package(evidence, kg, prefix='RESULT')
