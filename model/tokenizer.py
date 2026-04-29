import torch
from transformers import AddedToken, AutoTokenizer

from utils.condition import CONDITION_TOKENS
from utils.action_supervision import ACTION_CONTROL_TOKENS
from utils.evidence import EVIDENCE_CONTROL_TOKENS
from utils.logic_dsl import DSL_CONTROL_TOKENS
from utils.textualization import (
    GRAPH_TEXT_TOKENS,
    HYPOTHESIS_STRUCTURE_TOKENS,
    is_relation_text_token,
)


DEFAULT_CONDITION_TOKENS = CONDITION_TOKENS
TEXT_EXTRA_TOKENS = [
    'OBS',
    *DEFAULT_CONDITION_TOKENS,
    *HYPOTHESIS_STRUCTURE_TOKENS,
    *DSL_CONTROL_TOKENS,
    *ACTION_CONTROL_TOKENS,
    *EVIDENCE_CONTROL_TOKENS,
]


def get_text_extra_tokens(include_graph_tokens: bool = False):
    extra_tokens = list(TEXT_EXTRA_TOKENS)
    if include_graph_tokens:
        extra_tokens.extend(GRAPH_TEXT_TOKENS)
    return extra_tokens


def create_text_tokenizer(
        pretrained_model_path: str,
        extra_tokens=None,
        closed_text_tokens=None,
        trust_remote_code: bool = False):
    if extra_tokens is None:
        extra_tokens = get_text_extra_tokens(include_graph_tokens=False)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<|pad|>'
    special_tokens['sep_token'] = 'SEP'
    tokenizer.add_special_tokens(special_tokens)

    seen_tokens = set()
    tokens_to_add = []

    def maybe_add(token):
        if token == 'SEP' or token in seen_tokens:
            return
        seen_tokens.add(token)
        tokens_to_add.append(AddedToken(token, lstrip=False, rstrip=False, normalized=False))

    for token in sorted(set(extra_tokens)):
        maybe_add(token)

    if closed_text_tokens is not None:
        for token in sorted(set(closed_text_tokens)):
            maybe_add(token)

    tokenizer.add_tokens(tokens_to_add)
    return tokenizer, len(tokenizer)


def decode_text_token_ids(tokenizer, token_ids, preserve_whitespace: bool = False):
    special_ids = {
        token_id
        for token_id in [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
        ]
        if token_id is not None
    }
    filtered_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    decoded = tokenizer.decode(
        filtered_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if preserve_whitespace:
        return decoded.strip()
    return ' '.join(decoded.split())


def build_conditioned_source(source, condition_text=None):
    if condition_text is None:
        condition_text = [''] * len(source)

    merged_source = []
    for src, cond in zip(source, condition_text):
        cond = str(cond).strip()
        parts = [src]
        if cond:
            parts.extend(['SEP', cond])
        merged_source.append(' '.join(parts))
    return merged_source


def append_eos_to_targets(target, tokenizer):
    eos_token = tokenizer.eos_token
    if eos_token is None:
        return target
    return [
        text if str(text).rstrip().endswith(eos_token) else f'{text} {eos_token}'
        for text in target
    ]


def extract_text_sample_to_device(
        device,
        sample,
        tokenizer,
        src_len,
        tgt_len,
        is_gen: bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    condition_text = sample.get('condition_text', [''] * len(source))
    merged_source = build_conditioned_source(source, condition_text)
    target_with_eos = append_eos_to_targets(target, tokenizer)

    source_target_tokenized = tokenizer(
        merged_source,
        target_with_eos,
        padding='longest',
        return_tensors='pt',
    ).to(device)
    labels = torch.clone(source_target_tokenized.input_ids)

    source_tokenized = tokenizer(
        merged_source,
        padding='max_length',
        max_length=labels.shape[-1],
        return_tensors='pt',
    ).to(device)
    labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    if not is_gen:
        input_ids = source_target_tokenized.input_ids
        attention_mask = source_target_tokenized.attention_mask
    else:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        source_tokenized = tokenizer(
            merged_source,
            padding='longest',
            max_length=src_len,
            return_tensors='pt',
        ).to(device)
        tokenizer.padding_side = original_padding_side
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, condition_text


def new_extract_sample_to_device(device, sample, tokenizer, src_len, tgt_len, is_gen: bool):
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
    )


def new_extract_sample_to_device_condition(
        device,
        sample,
        tokenizer,
        src_len,
        tgt_len,
        is_gen: bool,
        condition_key: str = 'condition_text'):
    if condition_key in sample and condition_key != 'condition_text':
        sample = dict(sample)
        sample['condition_text'] = sample[condition_key]
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
    )


def new_extract_sample_to_device_pattern(device, sample, tokenizer, src_len, tgt_len, is_gen: bool):
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
    )
