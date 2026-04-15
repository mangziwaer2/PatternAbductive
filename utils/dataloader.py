import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from utils.load import load_sampled_dataset
from utils.parsing import ans_shift_indices, list_to_str, qry_shift_indices, qry_str_2_actionstr
from utils.textualization import observation_to_text, query_wordlist_to_text, query_wordlist_to_graph_text


REPRESENTATION_ID = 'id'
REPRESENTATION_TEXT = 'text'


def new_create_dataloader(dataset_dict, batch_size:int, drop_last:bool=False, shuffle:bool=True) :
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


def prepare_id_source_target(df: pd.DataFrame):
    source = df['answers'].apply(ans_shift_indices)
    source = source.apply(list_to_str)
    target = df['query'].apply(qry_shift_indices)
    target = target.apply(list_to_str)
    target = target.apply(qry_str_2_actionstr)
    return source, target


def prepare_text_source_target(
        df: pd.DataFrame,
        kg=None,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text'):
    def build_text_series(field_name: str):
        if field_name in df:
            return df[field_name].astype(str)
        if kg is None:
            raise KeyError(f'Missing required text field "{field_name}" and kg is not provided for fallback conversion.')
        if field_name == 'observation_text':
            return df['answers'].apply(lambda answers: observation_to_text(answers, kg))
        if field_name == 'hypothesis_text':
            return df['query'].apply(lambda query: query_wordlist_to_text(query, kg))
        if field_name == 'hypothesis_graph_text':
            return df['query'].apply(lambda query: query_wordlist_to_graph_text(query, kg))
        raise KeyError(f'Unsupported fallback text field: {field_name}')

    source = build_text_series(source_text_field)
    target = build_text_series(target_text_field)
    return source, target


def pre_processing(
        data: dict,
        pattern_str_2_id: dict,
        kg=None,
        representation: str = REPRESENTATION_ID,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text'):
    df = pd.DataFrame.from_records(data)
    if representation == REPRESENTATION_TEXT:
        source, target = prepare_text_source_target(
            df,
            kg=kg,
            source_text_field=source_text_field,
            target_text_field=target_text_field,
        )
    else:
        source, target = prepare_id_source_target(df)
    pattern_id = df['pattern_str'].apply(lambda x: pattern_str_2_id[x])
    nrows = len(df)

    def get_column_or_default(name, default):
        if name in df:
            return df[name].apply(lambda value: default if pd.isna(value) else value)
        return pd.Series([default] * nrows)

    return pd.concat({
        'source': source,
        'target': target,
        'pattern_id': pattern_id,
        'condition_text': get_column_or_default('condition_text', ''),
        'condition_text_textual': get_column_or_default('condition_text_textual', ''),
        'condition_signature': get_column_or_default('condition_signature', 'unconditional'),
        'condition_size': get_column_or_default('condition_size', 0),
        'condition_pattern': get_column_or_default('condition_pattern', ''),
        'condition_entity_number': get_column_or_default('condition_entity_number', -1),
        'condition_relation_number': get_column_or_default('condition_relation_number', -1),
        'condition_specific_entity': get_column_or_default('condition_specific_entity', -1),
        'condition_specific_relation': get_column_or_default('condition_specific_relation', 0),
        'query_entity_number': get_column_or_default('query_entity_number', -1),
        'query_relation_number': get_column_or_default('query_relation_number', -1),
        'query_unique_entity_number': get_column_or_default('query_unique_entity_number', -1),
        'query_unique_relation_number': get_column_or_default('query_unique_relation_number', -1),
        'observation_text': get_column_or_default('observation_text', ''),
        'hypothesis_text': get_column_or_default('hypothesis_text', ''),
        'hypothesis_graph_text': get_column_or_default('hypothesis_graph_text', ''),
    }, axis=1)

def new_create_dataset(dataname,
        pattern_filtered,
        data_root,
        splits,
        max_rows_by_split=None,
        kg=None,
        source_text_field: str = 'observation_text',
        target_text_field: str = 'hypothesis_text',
        representation: str = REPRESENTATION_ID,
        ):

    pattern_str_2_id = dict(zip(pattern_filtered['pattern_str'], pattern_filtered.index))

    data_dict, nentity, nrelation = load_sampled_dataset(
        data_root=data_root,
        dataname=dataname,
        splits=splits,
        max_rows_by_split=max_rows_by_split,
    )

    dataset_dict = {}
    for split in splits:
        df = pre_processing(
            data=data_dict[split],
            pattern_str_2_id=pattern_str_2_id,
            kg=kg,
            source_text_field=source_text_field,
            target_text_field=target_text_field,
            representation=representation)
        dataset_dict[split] = Dataset.from_pandas(df, split=split, preserve_index=False)

    return dataset_dict, nentity, nrelation
