import torch
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from transformers import AddedToken
from transformers import GPT2TokenizerFast

from utils.condition import CONDITION_TOKENS
from utils.textualization import (
    HYPOTHESIS_STRUCTURE_TOKENS,
    GRAPH_TEXT_TOKENS,
    is_relation_text_token,
)


DEFAULT_CONDITION_TOKENS = CONDITION_TOKENS
TEXT_EXTRA_TOKENS = ['OBS', *DEFAULT_CONDITION_TOKENS, *HYPOTHESIS_STRUCTURE_TOKENS]


def get_text_extra_tokens(include_graph_tokens: bool = False):
    extra_tokens = list(TEXT_EXTRA_TOKENS)
    if include_graph_tokens:
        extra_tokens.extend(GRAPH_TEXT_TOKENS)
    return extra_tokens

def number_to_pattern(input_str):
    elements = input_str.split()

    result = []
    for elem in elements:
        if elem.lstrip('-').isdigit():  # 检查是否是数字（包括负数）
            num = int(elem)
            if num < 0:
                result.append('p')  # 负数变为 p
            else:
                result.append('e')  # 正数变为 e
        else:
            result.append(elem)  # 非数字保持不变

    output_str = ' '.join(result)
    return output_str


def new_extract_sample_to_device(device,
                                 sample, tokenizer,
                                 src_len, tgt_len, is_gen: bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    source_target_tokenized = tokenizer(
        source, target,
        padding='longest',
        # max_length=src_len+tgt_len,
        return_tensors="pt").to(device)
    # labels is the source SEP target END, ...
    labels = torch.clone(source_target_tokenized.input_ids)

    source_tokenized = tokenizer(
        source,
        padding='max_length',
        max_length=labels.shape[-1],
        return_tensors="pt").to(device)
    labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    if is_gen == False:
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
    else:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        source_tokenized = tokenizer(
            source,
            padding='longest',
            max_length=src_len,
            return_tensors="pt").to(device)
        tokenizer.padding_side = original_padding_side
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target

def get_vocab(special_tokens, offset, nentity, nrelation, condition_tokens=None):
    if condition_tokens is None:
        condition_tokens = DEFAULT_CONDITION_TOKENS

    vocab = {}
    vocab.update(special_tokens)
    for i in range(1, nentity+1): # [offset, offset + nentity - 1]
        vocab[str(i)] = offset + i - 1
    for i in range(1, nrelation+1): # [offset + nentity, offset + nentity + nrelation - 1]
        vocab[str(-i)] = offset + nentity + i - 1
    next_token_id = offset + nentity + nrelation
    for token in condition_tokens:
        if token in vocab:
            continue
        vocab[token] = next_token_id
        next_token_id += 1
    return vocab, next_token_id

def create_tokenizer(
        special_tokens: dict, offset: int,
        nentity: int, nrelation: int,
        condition_tokens=None):
    pre_tokenizer = WhitespaceSplit()
    vocab, vocab_size = get_vocab(
        special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
        condition_tokens=condition_tokens,
    )
    model = WordLevel(vocab, unk_token='UNK')
    post_processor = TemplateProcessing(
        single='$0 SEP',
        pair='$A SEP $B END',
        special_tokens=[('SEP', special_tokens['SEP']), ('END', special_tokens['END'])]
    )
    tokenizer = Tokenizer(model=model)

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor
    # Just to let the tokenizer know about special tokens
    tokenizer.add_special_tokens(['START', 'END', 'PAD', 'UNK', 'SEP'])
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):
        tokenizer = GPT2TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='PAD',
            unk_token='UNK',
            sep_token='SEP',
            ) # default padding side
        # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, vocab_size


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
        token_id for token_id in [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
        ]
        if token_id is not None
    }
    filtered_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    tokens = tokenizer.convert_ids_to_tokens(filtered_ids)
    added_tokens = set(tokenizer.get_added_vocab().keys())
    if tokens and all(
            token in ['(', ')', 'p', 'i', 'u', 'n', 'e'] or is_relation_text_token(token) or token in added_tokens
            for token in tokens):
        return ' '.join(tokens)
    decoded = tokenizer.decode(
        filtered_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if preserve_whitespace:
        return decoded.strip()
    return ' '.join(decoded.split())


def build_conditioned_source(source, condition_text):
    if condition_text is None:
        return source

    merged_source = []
    for src, cond in zip(source, condition_text):
        if cond is None:
            merged_source.append(src)
            continue
        cond = str(cond).strip()
        merged_source.append(src if cond == '' else f'{src} SEP {cond}')
    return merged_source


def source_to_prompt(example, args=None):
    condition_key = 'condition_text_textual'
    if args is not None:
        condition_key = getattr(args, 'condition_field', condition_key)
    condition_text = example.get(condition_key, '')
    prompt = build_conditioned_source([example['source']], [condition_text])[0]
    enriched = dict(example)
    enriched['prompt'] = prompt
    return enriched


def new_extract_sample_to_device_pattern(device, sample, tokenizer,src_len, tgt_len, is_gen: bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_pattern(tgt) for tgt in target]
    merged_source = build_conditioned_source(source, target_pattern)

    source_target_tokenized = tokenizer(
        merged_source, target,  # 使用合并后的 source
        padding='longest',
        return_tensors="pt").to(device)
    labels = torch.clone(source_target_tokenized.input_ids)

    # 忽略 source 部分的 loss
    source_tokenized = tokenizer(
        merged_source,
        padding='max_length',
        max_length=labels.shape[-1],
        return_tensors="pt").to(device)
    labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    if not is_gen:  # 训练/验证阶段
        input_ids = source_target_tokenized.input_ids
        attention_mask = source_target_tokenized.attention_mask
    else:  # 测试/生成阶段（左填充）
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        source_tokenized = tokenizer(
            merged_source,  # 使用合并后的 source
            padding='longest',
            max_length=src_len,
            return_tensors="pt").to(device)
        tokenizer.padding_side = original_padding_side
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern


def new_extract_sample_to_device_condition(
        device,
        sample,
        tokenizer,
        src_len,
        tgt_len,
        is_gen: bool,
        condition_key: str = 'condition_text'):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    condition_text = sample[condition_key] if condition_key in sample else [''] * len(source)
    merged_source = build_conditioned_source(source, condition_text)

    source_target_tokenized = tokenizer(
        merged_source, target,
        padding='longest',
        return_tensors='pt').to(device)
    labels = torch.clone(source_target_tokenized.input_ids)

    source_tokenized = tokenizer(
        merged_source,
        padding='max_length',
        max_length=labels.shape[-1],
        return_tensors='pt').to(device)
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
            return_tensors='pt').to(device)
        tokenizer.padding_side = original_padding_side
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, condition_text
