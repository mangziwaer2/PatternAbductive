from pathlib import Path
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel


GPT2_MODEL_PATH = str('gpt2')


def _path_if_local(path_value: str) -> str:
    path = Path(str(path_value)).expanduser()
    return str(path) if path.exists() else str(path_value)


def _local_model_dir_for_name(model_name: str) -> str:
    candidates = [
        Path(str(model_name)).expanduser(),
        Path(__file__).resolve().parents[1] / str(model_name),
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / 'config.json').exists():
            return str(candidate)
    return ''


def resolve_model_runtime_config(model_name: str, config_model: dict | None = None) -> dict:
    if not config_model:
        runtime_config = {}
    elif model_name in config_model:
        runtime_config = dict(config_model[model_name] or {})
    elif 'default' in config_model:
        runtime_config = dict(config_model['default'] or {})
    else:
        runtime_config = {}

    local_model_dir = _local_model_dir_for_name(model_name)
    if local_model_dir:
        pretrained_path = str(runtime_config.get('pretrained_model_path', '') or '')
        tokenizer_path = str(runtime_config.get('tokenizer_path', '') or '')
        if not pretrained_path or '/' in pretrained_path or not Path(pretrained_path).expanduser().exists():
            runtime_config['pretrained_model_path'] = local_model_dir
        if not tokenizer_path or '/' in tokenizer_path or not Path(tokenizer_path).expanduser().exists():
            runtime_config['tokenizer_path'] = local_model_dir

    for key in ['pretrained_model_path', 'tokenizer_path']:
        if key in runtime_config:
            runtime_config[key] = _path_if_local(runtime_config[key])

    return runtime_config


def get_tokenizer_path(model_runtime_config: dict | None = None) -> str:
    model_runtime_config = model_runtime_config or {}
    return str(
        model_runtime_config.get('tokenizer_path')
        or model_runtime_config.get('pretrained_model_path')
        or GPT2_MODEL_PATH
    )


def create_transformer(ntoken: int, special_tokens: dict,
        model_name: str,
        vocab_size: Optional[int] = None,
        use_pretrained_weights: bool = False,
        model_runtime_config: dict | None = None):
    model_runtime_config = model_runtime_config or {}
    architecture = str(model_runtime_config.get('architecture', 'gpt2')).lower()
    pretrained_model_path = _path_if_local(model_runtime_config.get('pretrained_model_path', GPT2_MODEL_PATH))
    trust_remote_code = bool(model_runtime_config.get('trust_remote_code', False))
    use_pretrained_weights = bool(use_pretrained_weights or model_runtime_config.get('use_pretrained_weights', False))
    config_overrides = dict(model_runtime_config.get('config_overrides') or {})
    target_vocab_size = vocab_size if vocab_size is not None else ntoken + 1

    common_config = {
        'pad_token_id': special_tokens['PAD'],
        'bos_token_id': special_tokens['START'],
        'eos_token_id': special_tokens['END'],
        'decoder_start_token_id': special_tokens['START'],
        }
    if not use_pretrained_weights:
        common_config['vocab_size'] = target_vocab_size
    common_config.update(config_overrides)
    # Create transformers
        # default = huggingface gpt2 = the smallest version of GPT-2, with 124M parameters.
    if architecture == 'gpt2':
        config = GPT2Config.from_pretrained(
            pretrained_model_path,
            **common_config
        )
        if use_pretrained_weights:
            transformer = GPT2LMHeadModel.from_pretrained(
                pretrained_model_path,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            transformer = GPT2LMHeadModel(config)
    else:
        config = AutoConfig.from_pretrained(
            pretrained_model_path,
            trust_remote_code=trust_remote_code,
            **common_config,
        )
        if use_pretrained_weights:
            transformer = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                config=config,
                trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=True,
            )
        else:
            transformer = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code,
            )
    transformer.resize_token_embeddings(target_vocab_size)
    # Add attributes
    transformer.model_name = model_name
    return transformer
