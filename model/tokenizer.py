import torch
from transformers import AddedToken, GPT2TokenizerFast

from utils.condition import CONDITION_TOKENS
from utils.kg_hints import build_kg_hints_text
from utils.textualization import (
    GRAPH_TEXT_TOKENS,
    HYPOTHESIS_STRUCTURE_TOKENS,
    KG_HINT_TOKENS,
    is_relation_text_token,
)


DEFAULT_CONDITION_TOKENS = CONDITION_TOKENS
TEXT_EXTRA_TOKENS = ['OBS', *KG_HINT_TOKENS, *DEFAULT_CONDITION_TOKENS, *HYPOTHESIS_STRUCTURE_TOKENS]


def get_text_extra_tokens(include_graph_tokens: bool = False):
    extra_tokens = list(TEXT_EXTRA_TOKENS)
    if include_graph_tokens:
        extra_tokens.extend(GRAPH_TEXT_TOKENS)
    return extra_tokens


def create_text_tokenizer(pretrained_model_path: str, extra_tokens=None, closed_text_tokens=None):
    if extra_tokens is None:
        extra_tokens = get_text_extra_tokens(include_graph_tokens=False)

    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_path)
    tokenizer.add_special_tokens({
        'pad_token': '<|pad|>',
        'sep_token': 'SEP',
    })

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


def build_conditioned_source(source, condition_text=None, kg_hints_text=None):
    if condition_text is None:
        condition_text = [''] * len(source)
    if kg_hints_text is None:
        kg_hints_text = [''] * len(source)

    merged_source = []
    for src, cond, hints in zip(source, condition_text, kg_hints_text):
        cond = str(cond).strip()
        hints = str(hints).strip()
        parts = [src]
        if cond:
            parts.extend(['SEP', cond])
        if hints:
            parts.extend(['SEP', hints])
        merged_source.append(' '.join(parts))
    return merged_source


def extract_text_sample_to_device(
        device,
        sample,
        tokenizer,
        src_len,
        tgt_len,
        is_gen: bool,
        kg_hints_text=None):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    condition_text = sample.get('condition_text', [''] * len(source))
    merged_source = build_conditioned_source(source, condition_text, kg_hints_text=kg_hints_text)

    source_target_tokenized = tokenizer(
        merged_source,
        target,
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


def source_to_prompt(example, args=None, kg=None, kg_hint_split: str = 'train'):
    condition_text = example.get('condition_text', '')
    kg_hints_text = example.get('kg_hints_text', '')
    if not kg_hints_text and args is not None and getattr(args, 'use_kg_hints', False) and kg is not None:
        kg_hints_text = build_kg_hints_text(
            observation_text=example['source'],
            kg=kg,
            condition_text=condition_text,
            graph_split=kg_hint_split,
            max_facts=getattr(args, 'kg_hints_max_facts', 8),
        )
    prompt = build_conditioned_source([example['source']], [condition_text], [kg_hints_text])[0]
    enriched = dict(example)
    enriched['prompt'] = prompt
    return enriched


def new_extract_sample_to_device(device, sample, tokenizer, src_len, tgt_len, is_gen: bool, kg_hints_text=None):
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
        kg_hints_text=kg_hints_text,
    )


def new_extract_sample_to_device_condition(
        device,
        sample,
        tokenizer,
        src_len,
        tgt_len,
        is_gen: bool,
        condition_key: str = 'condition_text',
        kg_hints_text=None):
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
        kg_hints_text=kg_hints_text,
    )


def new_extract_sample_to_device_pattern(device, sample, tokenizer, src_len, tgt_len, is_gen: bool, kg_hints_text=None):
    return extract_text_sample_to_device(
        device=device,
        sample=sample,
        tokenizer=tokenizer,
        src_len=src_len,
        tgt_len=tgt_len,
        is_gen=is_gen,
        kg_hints_text=kg_hints_text,
    )
