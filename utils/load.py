import os
from pathlib import Path
import sys
import json
import shutil

import pandas as pd
import pickle
from datasets import Dataset as HFDataset, load_from_disk
import pykeen.datasets as pk_datasets
import pykeen.utils as pk_utils
import networkx as nx
import torch
try:
    import transformers
except ImportError:
    transformers = None

from utils.kgclass import KG

import yaml


KG_CACHE_DIR = './metadata/kg_cache'
DATASET_CACHE_DIR = './dataset_cache'

def load_model(path,contents,return_huggingface_model=True,epoch=0,
               model=None, optimizer=None, scheduler=None):
    print(f'# Loading checkpoint (model) {path}')
    if contents=="model":
        if model is not None or optimizer is not None or scheduler is not None:
            print('# Error: cannot pass model, optimizer, scheduler with contents="model"')
            exit()
        checkpoint = torch.load(path,weights_only=False)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    elif contents == 'rlmodel':
        print(f'# Loading checkpoint (model) {path}')
        if model is not None or optimizer is not None or scheduler is not None:
            print('# Error: cannot pass model, optimizer, scheduler with contents="model"')
            exit()
        checkpoint = torch.load(path,weights_only=False)
        model = checkpoint['model']
        model.warnings_issued = {}  # 重新添加属性
        def dummy_add_model_tags(self, tags):
            pass
        model.add_model_tags = dummy_add_model_tags.__get__(model)  # 重新绑定方法
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        print(f'# Error: contents "{contents}" not supported')
        exit()
    last_epoch = checkpoint.get(str(epoch), checkpoint.get('epoch', epoch))
    if 'loss_log' in checkpoint.keys():
        loss_log = checkpoint['loss_log']
    else:
        loss_log = {'train': {}, 'valid': {}}
    if return_huggingface_model and (transformers is None or not isinstance(model, transformers.PreTrainedModel)):
        print('Yes, returning .transformer')
        model = model.transformer
    return model, optimizer, scheduler, last_epoch, loss_log

def resolve_dataset_cache_path(dataname, dataset_cache_root=None):
    cache_root = dataset_cache_root or DATASET_CACHE_DIR
    cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cache_root, dataname))
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    return cache_path


def sanitize_cache_component(value):
    return ''.join(
        char if str(char).isalnum() or char in ('-', '_', '.')
        else '_'
        for char in str(value)
    )


def build_processed_dataset_cache_key(representation, source_text_field='observation_text', target_text_field='hypothesis_text'):
    return '__'.join([
        f'repr-{sanitize_cache_component(representation)}',
        f'src-{sanitize_cache_component(source_text_field)}',
        f'tgt-{sanitize_cache_component(target_text_field)}',
    ])


def resolve_processed_dataset_cache_path(
        dataname,
        split,
        representation,
        source_text_field='observation_text',
        target_text_field='hypothesis_text',
        dataset_cache_root=None):
    cache_root = resolve_dataset_cache_path(dataname, dataset_cache_root=dataset_cache_root)
    cache_key = build_processed_dataset_cache_key(
        representation=representation,
        source_text_field=source_text_field,
        target_text_field=target_text_field,
    )
    return os.path.join(cache_root, 'processed', cache_key, split)


def load_saved_processed_dataset(
        dataname,
        split,
        representation,
        source_text_field='observation_text',
        target_text_field='hypothesis_text',
        dataset_cache_root=None):
    dataset_path = resolve_processed_dataset_cache_path(
        dataname=dataname,
        split=split,
        representation=representation,
        source_text_field=source_text_field,
        target_text_field=target_text_field,
        dataset_cache_root=dataset_cache_root,
    )
    if not os.path.exists(os.path.join(dataset_path, 'state.json')):
        return None
    return load_from_disk(dataset_path)


def save_processed_dataset_to_disk(
        dataset,
        dataname,
        split,
        representation,
        source_text_field='observation_text',
        target_text_field='hypothesis_text',
        dataset_cache_root=None,
        overwrite=False):
    dataset_path = resolve_processed_dataset_cache_path(
        dataname=dataname,
        split=split,
        representation=representation,
        source_text_field=source_text_field,
        target_text_field=target_text_field,
        dataset_cache_root=dataset_cache_root,
    )
    if os.path.exists(dataset_path):
        if not overwrite:
            raise FileExistsError(f'Processed dataset cache already exists: {dataset_path}')
        shutil.rmtree(dataset_path)
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dataset_path)
    return dataset_path


def normalize_json_record(record):
    normalized = dict(record)
    query = normalized.get('query')
    if isinstance(query, list):
        normalized['query'] = json.dumps(query, ensure_ascii=False)
    return normalized


def iter_jsonl_records(data_path, max_rows: int = 0):
    rows_read = 0
    with open(data_path, 'r', encoding='utf-8') as source_file:
        for line in source_file:
            line = line.strip()
            if not line:
                continue
            yield normalize_json_record(json.loads(line))
            rows_read += 1
            if max_rows is not None and max_rows > 0 and rows_read >= max_rows:
                break


def resolve_raw_dataset_cache_path(data_path, split, max_rows=0, dataset_cache_root=None, dataname=None):
    cache_dir = resolve_dataset_cache_path(dataname or 'default', dataset_cache_root=dataset_cache_root)
    row_tag = 'full' if max_rows is None or max_rows <= 0 else f'first{int(max_rows)}'
    dataset_name = sanitize_cache_component(Path(data_path).stem)
    return os.path.join(cache_dir, 'raw', f'{dataset_name}-{sanitize_cache_component(split)}-{row_tag}')


def load_jsonl_as_hf_dataset(data_path, split: str, dataset_cache_root=None, dataname=None, max_rows: int = 0):
    dataset_path = resolve_raw_dataset_cache_path(
        data_path=data_path,
        split=split,
        max_rows=max_rows,
        dataset_cache_root=dataset_cache_root,
        dataname=dataname,
    )
    cache_state_path = os.path.join(dataset_path, 'state.json')
    if os.path.exists(cache_state_path) and os.path.getmtime(cache_state_path) >= os.path.getmtime(data_path):
        return load_from_disk(dataset_path)

    cache_dir = resolve_dataset_cache_path(dataname or 'default', dataset_cache_root=dataset_cache_root)
    dataset = HFDataset.from_generator(
        iter_jsonl_records,
        gen_kwargs={
            'data_path': data_path,
            'max_rows': max_rows,
        },
        cache_dir=cache_dir,
        keep_in_memory=False,
    )
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dataset_path)
    return dataset


def resolve_sampled_dataset_path(data_root, dataname, split):
    candidates = [
        os.path.join(data_root, dataname, f'{dataname}-{split}-a2q.jsonl'),
        os.path.join(data_root, dataname, f'{dataname}-{split}-a2q.json'),
        os.path.join(data_root, f'{dataname}-{split}-a2q.jsonl'),
        os.path.join(data_root, f'{dataname}-{split}-a2q.json'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'Cannot find sampled dataset for split "{split}". Checked: {candidates}'
    )


def resolve_stats_path(data_root, dataname):
    candidates = [
        os.path.join(data_root, dataname, 'stats.txt'),
        os.path.join(data_root, 'stats.txt'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'Cannot find stats.txt for dataset "{dataname}". Checked: {candidates}'
    )

def load_sampled_dataset_stats(data_root, dataname):
    stats_path = resolve_stats_path(data_root=data_root, dataname=dataname)
    with open(stats_path) as f:
        lines = f.readlines()
        nentity = int(lines[0].split('\t')[-1])
        nrelation = int(lines[1].split('\t')[-1])
    return nentity, nrelation


def load_sampled_dataset(data_root, dataname,
                         splits=['train', 'valid', 'test'],
                         max_rows_by_split=None,
                         dataset_cache_root=None):
    if max_rows_by_split is None:
        max_rows_by_split = {}
    data_dict = {}
    for split in splits:
        data_path = resolve_sampled_dataset_path(data_root=data_root, dataname=dataname, split=split)
        max_rows = max_rows_by_split.get(split, 0)
        dataset = load_jsonl_as_hf_dataset(
            data_path,
            split=split,
            dataset_cache_root=dataset_cache_root,
            dataname=dataname,
            max_rows=max_rows,
        )
        data_dict[split] = dataset

    nentity, nrelation = load_sampled_dataset_stats(data_root=data_root, dataname=dataname)
    # data_dict['test'] = [
    #     # {"answers":[3454,6345,3018,20909,19824,2802,9397,5144,16510,23708,23678],"query":["(","i","(","n","(","p","(",-549,")","(","e","(",12994,")",")",")",")","(","p","(",-547,")","(","e","(",2618,")",")",")",")"],"pattern_str":"(i,(n,(p,(e))),(p,(e)))"},
    #     # {"answers":[8645,6511,11929,9818,21918,21471],"query":["(","p","(",-195,")","(","u","(","p","(",-269,")","(","e","(",9116,")",")",")","(","p","(",-194,")","(","e","(",9818,")",")",")",")",")"],"pattern_str":"(p,(u,(p,(e)),(p,(e))))"},
    #     {"answers":[18369,22403,23044,19272,5898,1932,6160,1553,5653,15063,16672,12579,1060,14438,1511,23273,15917,15918,1069,15533,15924,13495,15929,24314,24317,7615],"query":["(","i","(","i","(","n","(","p","(",-659,")","(","e","(",18646,")",")",")",")","(","p","(",-659,")","(","e","(",7020,")",")",")",")","(","p","(",-659,")","(","e","(",7020,")",")",")",")"],"pattern_str":"(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))"},
    #     # {"answers":[9280,4355,12101,11334,23182,8914,1108,8024,10908,9954,23590,11688,7275,23212,2732,9135,17584,9140],"query":["(","i","(","n","(","p","(",-202,")","(","p","(",0,")","(","e","(",1949,")",")",")",")",")","(","p","(",-567,")","(","e","(",8134,")",")",")",")"],"pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"},
    #     {"answers":[17410,13187,3781,4581,4968,15085,23121,1845,6870,19801,15068,20029,6527],"query":["(","p","(",-119,")","(","e","(",2916,")",")",")"],"pattern_str":"(p,(e))"}
    # ]
    return data_dict, nentity, nrelation

def load_yaml(filename: str):
    with open(filename, 'r') as f:
        obj = yaml.safe_load(f)
    return obj

def list_to_graph(edges):
    g = nx.MultiDiGraph()
    g.add_edges_from(edges)
    # print(g.in_edges(0, keys=True))
    # print(g.out_edges(0, keys=True))
    return g

def df_to_graph(df):
    # list of (u, v, k) tuples
    edges = list(df.itertuples(index=False, name=None))
    return list_to_graph(edges)

def update_inverse_edges(rel_id2name: dict, raw_df: pd.DataFrame):
    """
    遍历原始关系 ID，执行固定公式：
        原始边：新ID = 原始ID × 2
        逆向边：新ID = 原始ID × 2 + 1
    2. 关系名称重命名
        原始：+关系名
        逆向：-关系名
    例
        原始关系：0=出生地
        正向：ID=0×2=0，名称 =+出生地
        逆向：ID=0×2+1=1，名称 =-出生地
    """
    new_id2name = {}
    rel_id2inv  = {}
    for id, name in rel_id2name.items():
        new_id2name[id * 2] = f'+{name}'
        new_id2name[id * 2 + 1] = f'-{name}'
        rel_id2inv[id * 2] = id * 2 + 1
        rel_id2inv[id * 2 + 1] = id * 2
    new_df = {}
    for split, df in raw_df.items():
        df_inv = pd.DataFrame(data=df, copy=True)
        # inverse edges
        df_inv.loc[:, ['head_id', 'tail_id']] = (df_inv.loc[:, ['tail_id', 'head_id']].values)
        # reindex rel id
        df['relation_id'] = df['relation_id'].apply(lambda x: x * 2)
        df_inv['relation_id'] = df_inv['relation_id'].apply(lambda x: x * 2 + 1)
        df_all = df_concat([df, df_inv])
        new_df[split] = df_all.sort_values(by=['relation_id'])
    return new_id2name, rel_id2inv, new_df

def df_concat(df_list: list):
    return pd.concat(df_list, ignore_index=True)

def load_kg_common(reverse_edges_flag: bool, id_map_only: bool):
    ds = pk_datasets.DBpedia50(create_inverse_triples=False)

    num_ent = ds.num_entities
    num_rel = ds.num_relations
    ent_id2name = pk_utils.invert_mapping(ds.entity_to_id)
    rel_id2name = pk_utils.invert_mapping(ds.relation_to_id)
    rel_id2inv = {}

    print('# During loading raw kg:')
    raw_df = {}
    for split in ['training', 'validation', 'testing']:
        if id_map_only == True: continue #如果只需要id的映射字典，则跳过，后面会直接返回结果
        factory = ds.factory_dict[split]
        mapped_triples = factory.mapped_triples
        triples_df = factory.tensor_to_df(mapped_triples)[['head_id', 'tail_id', 'relation_id']]
        raw_df[split] = triples_df

    raw_df_all = df_concat([raw_df['training'], raw_df['validation'], raw_df['testing']])
    raw_df['training'] = raw_df_all.sample(frac=0.8, replace=False, random_state=42)
    raw_df_remaining = raw_df_all.drop(raw_df['training'].index)
    raw_df['validation'] = raw_df_remaining.sample(frac=0.5, replace=False, random_state=42)
    raw_df['testing'] = raw_df_remaining.drop(raw_df['validation'].index)

    if reverse_edges_flag == True:
        rel_id2name, rel_id2inv, raw_df = update_inverse_edges(rel_id2name, raw_df)
        num_rel *= 2

    print('# Sizes after adding inverse edges')
    print(raw_df['training'].shape)
    print(raw_df['validation'].shape)
    print(raw_df['testing'].shape)

    if id_map_only == True:#只需要id的映射字典
        return {
            'ent_id2name': ent_id2name,
            'rel_id2name': rel_id2name
        }
    # creating graphs
    our_df = {
        'train': raw_df['training'],
        'valid': df_concat([raw_df['training'], raw_df['validation']]),
        'test': df_concat([raw_df['training'], raw_df['validation'], raw_df['testing']]),
        'test_only': raw_df['testing']
    }
    graphs = {}
    for split, df in our_df.items():
        graphs[split] = df_to_graph(df)

    print('# Checking id ranges (in graphs)')
    print(f'ent id: {min(ent_id2name.keys()), max(ent_id2name.keys())}')
    print(f'rel id: {min(rel_id2name.keys()), max(rel_id2name.keys())}')
    return {
        'num_ent': num_ent,
        'num_rel': num_rel,
        'ent_id2name': ent_id2name,
        'rel_id2name': rel_id2name,
        'rel_id2inv': rel_id2inv,
        'graphs': graphs
    }

def load_kg_from_disk(input_path):
    # Backward compatibility for older pickle files that were created before
    # `kgclass.py` was moved under `utils/`.
    from utils import kgclass as kgclass_module
    sys.modules.setdefault('kgclass', kgclass_module)
    with open(input_path,'rb') as f:
        kg = pickle.load(f)
        print(f"# KG loaded from {input_path}")
        return kg


def resolve_kg_cache_path(dataname):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", KG_CACHE_DIR, f'{dataname}.pkl'))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path

def dump_kg(kg,output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path,'wb') as f:
        pickle.dump(kg, f)
        print(f"KG saved to {output_path}")
    return kg

def load_kg(dataname, reverse_edges_flag=True, id_map_only=False):
    print(f'loading {dataname}')
    raw_kg_dict = load_kg_common(
        reverse_edges_flag=reverse_edges_flag,
        id_map_only=id_map_only
    )
    if raw_kg_dict == None: return None

    path = resolve_kg_cache_path(dataname)
    legacy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", f'./sampled_data/{dataname}.pkl'))
    if not os.path.exists(path) and os.path.exists(legacy_path):
        kg = load_kg_from_disk(legacy_path)
        dump_kg(kg, path)
        return kg
    if os.path.exists(path):
        kg = load_kg_from_disk(path)
    else:
        kg = KG(
            num_ent=raw_kg_dict['num_ent'],
            num_rel=raw_kg_dict['num_rel'],
            ent_id2name=raw_kg_dict['ent_id2name'],
            rel_id2name=raw_kg_dict['rel_id2name'],
            rel_id2inv=raw_kg_dict['rel_id2inv'],
            graphs=raw_kg_dict['graphs']
        )
        dump_kg(kg, path)

    return kg
