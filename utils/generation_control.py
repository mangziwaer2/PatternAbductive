import torch

from model.tokenizer import decode_text_token_ids
from utils.action_supervision import ACTION_END_TAG
from utils.logic_dsl import DSL_END_TAG


def _apply_top_k_top_p(logits, top_k: int = 0, top_p: float = 1.0):
    filtered = logits
    if top_k is not None and int(top_k) > 0:
        top_k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(filtered, top_k)[0][..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float('-inf'))

    if top_p is not None and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > float(top_p)
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove_mask = torch.zeros_like(filtered, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        filtered = filtered.masked_fill(remove_mask, float('-inf'))
    return filtered


@torch.no_grad()
def stream_generate_until_tags(
        model,
        tokenizer,
        prompt: str,
        device,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_strings: list[str] | None = None):
    stop_strings = stop_strings or [ACTION_END_TAG, DSL_END_TAG]
    tokenized = tokenizer(prompt, return_tensors='pt').to(device)
    attention_mask = tokenized.attention_mask
    generated_ids = []
    past_key_values = None
    next_input_ids = tokenized.input_ids

    for _ in range(max(1, int(max_new_tokens))):
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = getattr(outputs, 'past_key_values', None)

        if do_sample:
            logits = logits / max(float(temperature), 1e-6)
            logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = int(next_token.item())
        generated_ids.append(token_id)
        completion = decode_text_token_ids(
            tokenizer,
            generated_ids,
            preserve_whitespace=True,
        )

        if token_id == tokenizer.eos_token_id:
            break
        if any(stop_string in completion for stop_string in stop_strings):
            break

        next_input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

    return {
        'prompt': prompt,
        'generated_ids': generated_ids,
        'completion': decode_text_token_ids(tokenizer, generated_ids, preserve_whitespace=True),
    }
