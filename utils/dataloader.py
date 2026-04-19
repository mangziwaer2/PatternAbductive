import json

from torch.utils.data import DataLoader

from utils.load import load_sampled_dataset, load_sampled_dataset_stats, load_saved_processed_dataset
from utils.parsing import ans_shift_indices, list_to_str, qry_shift_indices, qry_str_2_actionstr
from utils.textualization import observation_to_text, query_wordlist_to_text, query_wordlist_to_graph_text


REPRESENTATION_ID = 'id'
REPRESENTATION_TEXT = 'text'


def new_create_dataloader(dataset_dict, batch_size: int, drop_last: bool = False, shuffle: bool = True):
    import warnings
    if drop_last:
        warnings.warn('drop_last is True')
    dataloader_dict = {}
    for split, dataset in dataset_dict.items():
        dataloader_dict[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4
        )
    return dataloader_dict


def _normalize_value(value, default):
    if value is None:
        return default
    try:
        if value != value:
            return default
    except Exception:
        pass
    return value


def _get_batch_size(batch):
    if not batch:
        return 0
    first_key = next(iter(batch))
    return len(batch[first_key])


def _get_column_or_default(batch, name, default):
    batch_size = _get_batch_size(batch)
    if name not in batch:
        return [default] * batch_size
    return [_normalize_value(value, default) for value in batch[name]]


def _select_dataset_rows(dataset, max_rows):
    if max_rows is None or max_rows <= 0:
        return dataset
    nrows = min(len(dataset), max_rows)
    return dataset.select(range(nrows))


def filter_dataset_by_excluded_condition_types(dataset, excluded_condition_types=None):
    if not excluded_condition_types or 'condition_signature' not in dataset.column_names:
        return dataset

    excluded_condition_types = {
        str(condition_type).strip()
        for condition_type in excluded_condition_types
        if str(condition_type).strip() != ''
    }
    if not excluded_condition_types:
        return dataset

    def keep_example(example):
        signature = str(example.get('condition_signature', 'unconditional') or 'unconditional')
        if signature == 'unconditional':
            return True
        present_types = {token for token in signature.split('+') if token}
        return len(present_types.intersection(excluded_condition_types)) == 0

    return dataset.filter(keep_example, load_from_cache_file=False)


def _decode_query_value(query):
    if isinstance(query, str):
        query = query.strip()
        if query.startswith('[') and query.endswith(']'):
            return json.loads(query)
    return query


def prepare_id_source_target_batch(batch):
    source = [list_to_str(ans_shift_indices(answers)) for answers in batch['answers']]
    target = [qry_str_2_actionstr(list_to_str(qry_shift_indices(_decode_query_value(query)))) for query in batch['query']]
    return source, target


def prepare_text_source_target_batch(
        batch,
        kg=None,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text'):
    def build_text_column(field_name: str):
        if field_name in batch:
            return [str(value) for value in batch[field_name]]
        if kg is None:
            raise KeyError(f'Missing required text field "{field_name}" and kg is not provided for fallback conversion.')
        if field_name == 'observation_text':
            return [observation_to_text(answers, kg) for answers in batch['answers']]
        if field_name == 'hypothesis_text':
            return [query_wordlist_to_text(_decode_query_value(query), kg) for query in batch['query']]
        if field_name == 'hypothesis_graph_text':
            return [query_wordlist_to_graph_text(_decode_query_value(query), kg) for query in batch['query']]
        raise KeyError(f'Unsupported fallback text field: {field_name}')

    source = build_text_column(source_text_field)
    target = build_text_column(target_text_field)
    return source, target


def preprocess_batch(
        batch,
        pattern_str_2_id: dict,
        kg=None,
        representation: str = REPRESENTATION_ID,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text'):
    if representation == REPRESENTATION_TEXT:
        source, target = prepare_text_source_target_batch(
            batch,
            kg=kg,
            source_text_field=source_text_field,
            target_text_field=target_text_field,
        )
    else:
        source, target = prepare_id_source_target_batch(batch)

    return {
        'source': source,
        'target': target,
        'pattern_id': [pattern_str_2_id[pattern_str] for pattern_str in batch['pattern_str']],
        'condition_text': _get_column_or_default(batch, 'condition_text', ''),
        'condition_text_textual': _get_column_or_default(batch, 'condition_text_textual', ''),
        'condition_signature': _get_column_or_default(batch, 'condition_signature', 'unconditional'),
        'condition_size': _get_column_or_default(batch, 'condition_size', 0),
        'condition_pattern': _get_column_or_default(batch, 'condition_pattern', ''),
        'condition_entity_number': _get_column_or_default(batch, 'condition_entity_number', -1),
        'condition_relation_number': _get_column_or_default(batch, 'condition_relation_number', -1),
        'condition_specific_entity': _get_column_or_default(batch, 'condition_specific_entity', -1),
        'condition_specific_relation': _get_column_or_default(batch, 'condition_specific_relation', 0),
        'query_entity_number': _get_column_or_default(batch, 'query_entity_number', -1),
        'query_relation_number': _get_column_or_default(batch, 'query_relation_number', -1),
        'query_unique_entity_number': _get_column_or_default(batch, 'query_unique_entity_number', -1),
        'query_unique_relation_number': _get_column_or_default(batch, 'query_unique_relation_number', -1),
        'observation_text': _get_column_or_default(batch, 'observation_text', ''),
        'hypothesis_text': _get_column_or_default(batch, 'hypothesis_text', ''),
        'hypothesis_graph_text': _get_column_or_default(batch, 'hypothesis_graph_text', ''),
    }


def new_create_dataset(dataname,
        pattern_filtered,
        data_root,
        splits,
        max_rows_by_split=None,
        kg=None,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text',
        representation: str = REPRESENTATION_ID,
        dataset_cache_root: str = None,
        dataset_num_proc: int = 1,
        dataset_map_batch_size: int = 1000,
        prefer_saved_processed_cache: bool = True):

    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered.index))
    nentity, nrelation = load_sampled_dataset_stats(data_root=data_root, dataname=dataname)

    dataset_dict = {}
    splits_to_process = []
    for split in splits:
        cached_dataset = None
        if prefer_saved_processed_cache:
            cached_dataset = load_saved_processed_dataset(
                dataname=dataname,
                split=split,
                representation=representation,
                source_text_field=source_text_field,
                target_text_field=target_text_field,
                dataset_cache_root=dataset_cache_root,
            )
        if cached_dataset is None:
            splits_to_process.append(split)
            continue
        dataset_dict[split] = _select_dataset_rows(
            cached_dataset,
            (max_rows_by_split or {}).get(split, 0),
        )

    data_dict = {}
    if splits_to_process:
        data_dict, _, _ = load_sampled_dataset(
            data_root=data_root,
            dataname=dataname,
            splits=splits_to_process,
            max_rows_by_split=max_rows_by_split,
            dataset_cache_root=dataset_cache_root,
        )

    for split in splits_to_process:
        raw_dataset = data_dict[split]
        needs_kg = (
            representation == REPRESENTATION_TEXT and (
                source_text_field not in raw_dataset.column_names
                or target_text_field not in raw_dataset.column_names
            )
        )
        map_kwargs = {
            'function': preprocess_batch,
            'fn_kwargs': {
                'pattern_str_2_id': pattern_str_2_id,
                'kg': kg if needs_kg else None,
                'representation': representation,
                'source_text_field': source_text_field,
                'target_text_field': target_text_field,
            },
            'batched': True,
            'batch_size': dataset_map_batch_size,
            'remove_columns': raw_dataset.column_names,
            'load_from_cache_file': True,
            'keep_in_memory': False,
            'desc': f'preprocess_{split}',
        }
        if dataset_num_proc is not None and dataset_num_proc > 1:
            map_kwargs['num_proc'] = dataset_num_proc
        dataset_dict[split] = raw_dataset.map(**map_kwargs)

    return dataset_dict, nentity, nrelation
