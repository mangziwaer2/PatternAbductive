from pathlib import Path
from typing import Optional

from transformers import GPT2Config, GPT2LMHeadModel


GPT2_MODEL_PATH = str('gpt2')


def create_transformer(ntoken: int, special_tokens: dict,
        model_name: str,
        vocab_size: Optional[int] = None,
        use_pretrained_weights: bool = False):

    common_config = {
        'vocab_size': vocab_size if vocab_size is not None else ntoken + 1, # pad_token is negative
        'pad_token_id': special_tokens['PAD'],
        'bos_token_id': special_tokens['START'],
        'eos_token_id': special_tokens['END'],
        'decoder_start_token_id': special_tokens['START'],
        'n_layer': 6,
        }
    # Create transformers
        # default = huggingface gpt2 = the smallest version of GPT-2, with 124M parameters.
    config = GPT2Config.from_pretrained(
        GPT2_MODEL_PATH,
        **common_config
    )
    if use_pretrained_weights:
        transformer = GPT2LMHeadModel.from_pretrained(
            GPT2_MODEL_PATH,
            config=config,
            ignore_mismatched_sizes=True,
        )
        transformer.resize_token_embeddings(config.vocab_size)
    else:
        transformer = GPT2LMHeadModel(config)
    # Add attributes
    transformer.model_name = model_name
    return transformer
