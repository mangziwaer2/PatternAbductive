import argparse
import json
import os
import random

from utils.load import load_kg, load_kg_from_disk, resolve_kg_cache_path, resolve_sampled_dataset_path
from utils.parsing import qry_wordlist_2_nestedlist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--data_root', default='./sampled_data_compact/')
    parser.add_argument('--split', default='train')
    parser.add_argument('--index', type=int, help='0-based row index to inspect')
    parser.add_argument('-n', '--num-samples', type=int, default=3)
    parser.add_argument('--random', action='store_true', help='Sample rows uniformly at random')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show-names', action='store_true', help='Resolve entity/relation ids to names')
    parser.add_argument('--max-answers', type=int, default=12)
    return parser.parse_args()


def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_row_by_index(path, index):
    for i, row in enumerate(iter_jsonl(path)):
        if i == index:
            return i, row
    raise IndexError(f'Index {index} is out of range for {path}')


def reservoir_sample_rows(path, num_samples, seed):
    rng = random.Random(seed)
    reservoir = []
    for index, row in enumerate(iter_jsonl(path)):
        item = (index, row)
        if index < num_samples:
            reservoir.append(item)
            continue
        replacement_index = rng.randint(0, index)
        if replacement_index < num_samples:
            reservoir[replacement_index] = item
    return sorted(reservoir, key=lambda item: item[0])


def get_first_rows(path, num_samples):
    rows = []
    for index, row in enumerate(iter_jsonl(path)):
        rows.append((index, row))
        if len(rows) >= num_samples:
            break
    return rows


def load_kg_for_preview(dataname):
    pkl_path = resolve_kg_cache_path(dataname)
    if os.path.exists(pkl_path):
        return load_kg_from_disk(pkl_path)
    return load_kg(dataname)


def entity_name(kg, entity_id):
    if kg is None:
        return None
    return kg.ent_id2name.get(entity_id, '<unknown-entity>')


def relation_name(kg, relation_id):
    if kg is None:
        return None
    return kg.rel_id2name.get(-relation_id, '<unknown-relation>')


def format_entity(entity_id, kg):
    name = entity_name(kg, entity_id)
    return str(entity_id) if name is None else f'{entity_id} [{name}]'


def format_relation(relation_id, kg):
    name = relation_name(kg, relation_id)
    return str(relation_id) if name is None else f'{relation_id} [{name}]'


def format_id_list(values, formatter, limit):
    values = list(values)
    shown = values[:limit]
    text = ', '.join(formatter(value) for value in shown)
    if len(values) > limit:
        text += f', ... (+{len(values) - limit} more)'
    return text if text else '<empty>'


def format_condition_block(row, kg):
    condition_pairs = []
    if row.get('condition_pattern') is not None:
        condition_pairs.append(('pattern', row['condition_pattern']))
    if row.get('condition_entity_number') is not None:
        condition_pairs.append(('entity-number', row['condition_entity_number']))
    if row.get('condition_relation_number') is not None:
        condition_pairs.append(('relation-number', row['condition_relation_number']))
    if row.get('condition_specific_entity') is not None:
        condition_pairs.append(('specific-entity', format_entity(row['condition_specific_entity'], kg)))
    if row.get('condition_specific_relation') is not None:
        condition_pairs.append(('specific-relation', format_relation(row['condition_specific_relation'], kg)))
    if not condition_pairs:
        return ['  type: unconditional']
    return [f'  {key}: {value}' for key, value in condition_pairs]


def format_query_tree(query_nested, kg, indent=0):
    pad = '  ' * indent
    operator, *args = query_nested

    if operator == 'e':
        [[entity_id]] = args
        return [f'{pad}e {format_entity(entity_id, kg)}']

    if operator == 'p':
        [relation_id], sub_query = args
        lines = [f'{pad}p {format_relation(relation_id, kg)}']
        lines.extend(format_query_tree(sub_query, kg, indent + 1))
        return lines

    if operator == 'n':
        lines = [f'{pad}n']
        lines.extend(format_query_tree(args[0], kg, indent + 1))
        return lines

    if operator in ['i', 'u']:
        lines = [f'{pad}{operator}']
        for sub_query in args:
            lines.extend(format_query_tree(sub_query, kg, indent + 1))
        return lines

    return [f'{pad}{operator}']


def render_sample(row_index, row, kg, max_answers):
    query_wordlist = list(row['query'])
    query_nested = qry_wordlist_2_nestedlist(query_wordlist[:])
    query_text = ' '.join(str(token) for token in query_wordlist)

    parts = []
    parts.append('=' * 80)
    parts.append(f'Row {row_index}')
    parts.append(f'base_sample_id: {row.get("base_sample_id")}')
    parts.append('')
    parts.append('Observation')
    parts.append(f'  answer_count: {len(row["answers"])}')
    parts.append(f'  answers: {format_id_list(row["answers"], lambda value: format_entity(value, kg), max_answers)}')
    if row.get('observation_text'):
        parts.append(f'  observation_text: {row["observation_text"]}')
    parts.append('')
    parts.append('Condition')
    parts.append(f'  signature: {row.get("condition_signature", "unconditional")}')
    parts.append(f'  text: {row.get("condition_text") or "<empty>"}')
    if row.get('condition_text_textual'):
        parts.append(f'  text_textual: {row["condition_text_textual"]}')
    parts.extend(format_condition_block(row, kg))
    parts.append('')
    parts.append('Hypothesis')
    parts.append(f'  pattern: {row.get("pattern_str")}')
    parts.append(
        f'  counts: entities={row.get("query_entity_number")} '
        f'(unique={row.get("query_unique_entity_number")}), '
        f'relations={row.get("query_relation_number")} '
        f'(unique={row.get("query_unique_relation_number")})'
    )
    parts.append(
        f'  anchor_entities: '
        f'{format_id_list(row.get("query_anchor_entities", []), lambda value: format_entity(value, kg), max_answers)}'
    )
    parts.append(
        f'  relations: '
        f'{format_id_list(row.get("query_relations", []), lambda value: format_relation(value, kg), max_answers)}'
    )
    parts.append(f'  raw_query: {query_text}')
    if row.get('hypothesis_text'):
        parts.append(f'  hypothesis_text: {row["hypothesis_text"]}')
    if row.get('hypothesis_graph_text'):
        parts.append('  hypothesis_graph_text:')
        parts.extend(f'    {line}' for line in str(row['hypothesis_graph_text']).splitlines())
    parts.append('  tree:')
    if query_nested is None:
        parts.append('    <failed to parse query>')
    else:
        parts.extend(format_query_tree(query_nested, kg, indent=2))
    return '\n'.join(parts)


def main():
    args = parse_args()
    dataset_path = resolve_sampled_dataset_path(
        data_root=args.data_root,
        dataname=args.dataname,
        split=args.split,
    )

    kg = load_kg_for_preview(args.dataname) if args.show_names else None

    if args.index is not None:
        rows = [get_row_by_index(dataset_path, args.index)]
    elif args.random:
        rows = reservoir_sample_rows(dataset_path, args.num_samples, args.seed)
    else:
        rows = get_first_rows(dataset_path, args.num_samples)

    for idx, row in rows:
        print(render_sample(idx, row, kg, args.max_answers))


if __name__ == '__main__':
    main()
