import json

from torch.utils.data import DataLoader

from utils.load import (
    load_sampled_dataset,
    load_sampled_dataset_stats,
    load_saved_processed_dataset,
    save_processed_dataset_to_disk,
)
from utils.logic_dsl import pattern_str_to_pattern_dsl, surface_query_to_dsl, tag_dsl_text
from utils.evidence import RESULT_END_TAG, RESULT_START_TAG
from utils.text_dataset import build_stage2_trace, result_has_edges
from utils.action_supervision import ACTION_END_TAG, ACTION_START_TAG, strip_action_tags


REPRESENTATION_TEXT = 'text'
DEFAULT_TRAIN_STAGE = 'logic'
ACTION_SCHEMA_VERSION = 'actionv5_tagged_io'
CURRENT_ACTION_PREFIXES = (
    'ACTION FIND_COMMON ',
    'ACTION FIND_ALTERNATIVE ',
    'ACTION FIND_EXCLUSION ',
    'ACTION EXPAND ',
)


def new_create_dataloader(
        dataset_dict,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2):
    import warnings

    if drop_last:
        warnings.warn('drop_last is True')
    dataloader_dict = {}
    for split, dataset in dataset_dict.items():
        worker_count = max(0, int(num_workers))
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'drop_last': drop_last,
            'num_workers': worker_count,
            'pin_memory': bool(pin_memory),
        }
        if worker_count > 0:
            dataloader_kwargs['persistent_workers'] = bool(persistent_workers)
            dataloader_kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
        dataloader_dict[split] = DataLoader(dataset, **dataloader_kwargs)
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


def _derive_logic_dsl_values(batch, kg):
    if 'logic_dsl' in batch:
        return _get_column_or_default(batch, 'logic_dsl', '')
    if kg is None:
        raise KeyError('Missing "logic_dsl" and kg is required to derive it from hypothesis_text.')
    return [surface_query_to_dsl(query_text, kg) for query_text in batch['hypothesis_text']]


def _derive_pattern_dsl_values(batch):
    if 'pattern_dsl' in batch:
        return _get_column_or_default(batch, 'pattern_dsl', '')
    return [pattern_str_to_pattern_dsl(pattern_str) for pattern_str in batch['pattern_str']]


def _normalize_nested_records(value):
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except Exception:
            return []
    if isinstance(value, dict):
        keys = list(value.keys())
        if not keys:
            return []
        first_value = value[keys[0]]
        if not isinstance(first_value, list):
            return [value]
        length = len(first_value)
        records = []
        for index in range(length):
            records.append({
                key: values[index] if isinstance(values, list) and index < len(values) else values
                for key, values in value.items()
            })
        return records
    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]
    return []


def trace_uses_current_action_schema(trace) -> bool:
    normalized = _normalize_nested_records(trace)
    if not normalized:
        return False
    action_count = 0
    for event in normalized:
        action = str(event.get('action', '')).strip()
        if not action:
            continue
        action_count += 1
        if ACTION_START_TAG not in action or ACTION_END_TAG not in action:
            return False
        action_body = strip_action_tags(action)
        if not any(action_body.startswith(prefix) for prefix in CURRENT_ACTION_PREFIXES):
            return False
        result = str(event.get('result', '')).strip()
        if RESULT_START_TAG not in result or RESULT_END_TAG not in result:
            return False
        if not result_has_edges(result):
            return False
    return action_count > 0


def record_has_current_stage2_trace(record) -> bool:
    return trace_uses_current_action_schema(record.get('stage2_trace'))


def _derive_stage2_traces_for_batch(batch, kg, graph_split, top_k):
    trace_values = _get_column_or_default(batch, 'stage2_trace', []) if 'stage2_trace' in batch else []
    traces = []
    missing_indices = []
    for index, trace in enumerate(trace_values):
        normalized = _normalize_nested_records(trace)
        if normalized and trace_uses_current_action_schema(normalized):
            traces.append(normalized)
        else:
            traces.append(None)
            missing_indices.append(index)

    if missing_indices and kg is None:
        raise ValueError(
            'Stage 2 requires current-schema stage2_trace or a loaded KG. '
            'Run sampling.py again to generate traces, or train with --force_load_kg to rebuild them lazily.'
        )

    if not trace_values:
        traces = [None] * len(batch['pattern_str'])
        missing_indices = list(range(len(traces)))

    for index in missing_indices:
        traces[index] = build_stage2_trace(
            pattern_str=batch['pattern_str'][index],
            observation_text=batch['observation_text'][index],
            kg=kg,
            graph_split=graph_split,
            top_k=top_k,
        )
    return traces


def _trace_to_prefix_samples(observation_text, trace, logic_dsl):
    steps = []
    history_parts = []
    for event in _normalize_nested_records(trace):
        action = str(event.get('action', '')).strip()
        result = str(event.get('result', '')).strip()
        if not action:
            continue
        steps.append({
            'source': '\n'.join([observation_text, *history_parts]),
            'target': action,
            'target_type': 'action',
        })
        if result:
            history_parts.extend([action, result])
        else:
            history_parts.append(action)

    steps.append({
        'source': '\n'.join([observation_text, *history_parts]),
        'target': tag_dsl_text(logic_dsl),
        'target_type': 'dsl',
    })
    return steps


def _build_stage2_prefix_samples_for_batch(batch, kg, graph_split, top_k):
    logic_dsl = _derive_logic_dsl_values(batch, kg=kg)
    traces = _derive_stage2_traces_for_batch(
        batch=batch,
        kg=kg,
        graph_split=graph_split,
        top_k=top_k,
    )
    return [
        _trace_to_prefix_samples(observation_text=obs, trace=trace, logic_dsl=logic)
        for obs, trace, logic in zip(batch['observation_text'], traces, logic_dsl)
    ]


def prepare_train_stage_source_target_batch(
        batch,
        kg=None,
        graph_split: str = 'train',
        train_stage: str = DEFAULT_TRAIN_STAGE,
        result_top_k: int = 10):
    observation = [str(value) for value in batch['observation_text']]

    if train_stage == 'logic':
        source = observation
        pattern_dsl = _derive_pattern_dsl_values(batch)
        logic_dsl = _derive_logic_dsl_values(batch, kg=kg)
        target = [
            f'PATTERN {pattern} {tag_dsl_text(logic)}'
            for pattern, logic in zip(pattern_dsl, logic_dsl)
        ]
        extra = _build_common_extra(batch, kg=kg, observation=observation, result_top_k=result_top_k)
        extra['target_type'] = ['pattern_dsl'] * len(source)
        return source, target, extra

    if train_stage == 'stage2':
        stage2_source = []
        stage2_target = []
        stage2_target_type = []
        stage2_pattern_str = []

        logic_dsl = _derive_logic_dsl_values(batch, kg=kg)
        prefix_samples_by_row = _build_stage2_prefix_samples_for_batch(
            batch=batch,
            kg=kg,
            graph_split=graph_split,
            top_k=result_top_k,
        )

        for pattern_str, steps in zip(
                batch['pattern_str'],
                prefix_samples_by_row):
            for step in steps:
                stage2_source.append(str(step.get('source', '')))
                stage2_target.append(str(step.get('target', '')))
                stage2_target_type.append(str(step.get('target_type', '')))
                stage2_pattern_str.append(pattern_str)

        extra = {
            'condition_text': [''] * len(stage2_source),
            'target_type': stage2_target_type,
            'pattern_str_values': stage2_pattern_str,
        }
        return stage2_source, stage2_target, extra

    raise ValueError(f'Unsupported train_stage: {train_stage}')


def _build_common_extra(batch, kg, observation, result_top_k):
    del kg, result_top_k
    return {
        'condition_text': [''] * len(observation),
        'pattern_str_values': batch['pattern_str'],
    }


def _select_dataset_rows(dataset, max_rows):
    if max_rows is None or max_rows <= 0:
        return dataset
    nrows = min(len(dataset), max_rows)
    return dataset.select(range(nrows))


def _cache_fields_for_stage(source_text_field, target_text_field, train_stage, result_top_k, max_rows):
    row_tag = 'full' if max_rows is None or max_rows <= 0 else f'first{int(max_rows)}'
    schema_tag = ACTION_SCHEMA_VERSION if train_stage == 'stage2' else 'logicv2'
    stage_tag = f'{train_stage}|{schema_tag}|topk{int(result_top_k)}|rows-{row_tag}'
    return f'{source_text_field}|{stage_tag}', f'{target_text_field}|{stage_tag}'


def _dataset_has_current_trace(dataset, sample_size: int = 1024):
    if 'stage2_trace' not in dataset.column_names:
        return False
    sample_count = min(len(dataset), max(1, int(sample_size)))
    for index in range(sample_count):
        if record_has_current_stage2_trace(dataset[index]):
            return True
    return False


def _stage_preprocess_needs_kg(raw_dataset, train_stage):
    needs_logic_dsl = 'logic_dsl' not in raw_dataset.column_names
    if train_stage == 'logic':
        return needs_logic_dsl
    if train_stage == 'stage2':
        return needs_logic_dsl or not _dataset_has_current_trace(raw_dataset)
    return False


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


def preprocess_batch(
        batch,
        pattern_str_2_id: dict,
        kg=None,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text',
        train_stage: str = DEFAULT_TRAIN_STAGE,
        graph_split: str = 'train',
        result_top_k: int = 10):
    del source_text_field, target_text_field
    source, target, extra = prepare_train_stage_source_target_batch(
        batch=batch,
        kg=kg,
        graph_split=graph_split,
        train_stage=train_stage,
        result_top_k=result_top_k,
    )
    condition_text = extra['condition_text']

    return {
        'source': source,
        'target': target,
        'pattern_id': [pattern_str_2_id[pattern_str] for pattern_str in extra.get('pattern_str_values', batch['pattern_str'])],
        'condition_text': condition_text,
        'target_type': extra.get('target_type', [''] * len(source)),
    }


def new_create_dataset(
        dataname,
        pattern_filtered,
        data_root,
        splits,
        max_rows_by_split=None,
        kg=None,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text',
        representation: str = REPRESENTATION_TEXT,
        dataset_cache_root: str = None,
        dataset_num_proc: int = 1,
        dataset_map_batch_size: int = 1000,
        prefer_saved_processed_cache: bool = True,
        train_stage: str = DEFAULT_TRAIN_STAGE,
        result_top_k: int = 10):
    if representation != REPRESENTATION_TEXT:
        raise ValueError('Only text representation is supported in the simplified pipeline.')

    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered.index))
    data_root_tag = data_root
    load_sampled_dataset_stats(data_root=data_root, dataname=dataname)

    dataset_dict = {}
    splits_to_process = []
    for split in splits:
        split_max_rows = (max_rows_by_split or {}).get(split, 0)
        cache_source_text_field, cache_target_text_field = _cache_fields_for_stage(
            source_text_field=source_text_field,
            target_text_field=target_text_field,
            train_stage=train_stage,
            result_top_k=result_top_k,
            max_rows=split_max_rows,
        )
        cached_dataset = None
        if prefer_saved_processed_cache:
            cached_dataset = load_saved_processed_dataset(
                dataname=dataname,
                split=split,
                representation=representation,
                source_text_field=cache_source_text_field,
                target_text_field=cache_target_text_field,
                data_root_tag=data_root_tag,
                dataset_cache_root=dataset_cache_root,
            )
        if cached_dataset is None:
            splits_to_process.append(split)
            continue
        dataset_dict[split] = _select_dataset_rows(
            cached_dataset,
            split_max_rows,
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
        split_max_rows = (max_rows_by_split or {}).get(split, 0)
        cache_source_text_field, cache_target_text_field = _cache_fields_for_stage(
            source_text_field=source_text_field,
            target_text_field=target_text_field,
            train_stage=train_stage,
            result_top_k=result_top_k,
            max_rows=split_max_rows,
        )
        needs_kg = _stage_preprocess_needs_kg(raw_dataset, train_stage)
        map_kwargs = {
            'function': preprocess_batch,
            'fn_kwargs': {
                'pattern_str_2_id': pattern_str_2_id,
                'kg': kg if needs_kg else None,
                'source_text_field': source_text_field,
                'target_text_field': target_text_field,
                'train_stage': train_stage,
                'graph_split': split,
                'result_top_k': result_top_k,
            },
            'batched': True,
            'batch_size': dataset_map_batch_size,
            'remove_columns': raw_dataset.column_names,
            'load_from_cache_file': False,
            'keep_in_memory': False,
            'desc': f'preprocess_{split}',
        }
        if dataset_num_proc is not None and dataset_num_proc > 1:
            map_kwargs['num_proc'] = dataset_num_proc
        dataset_dict[split] = raw_dataset.map(**map_kwargs)
        save_processed_dataset_to_disk(
            dataset=dataset_dict[split],
            dataname=dataname,
            split=split,
            representation=representation,
            source_text_field=cache_source_text_field,
            target_text_field=cache_target_text_field,
            data_root_tag=data_root_tag,
            dataset_cache_root=dataset_cache_root,
            overwrite=True,
        )

    return dataset_dict, None, None
