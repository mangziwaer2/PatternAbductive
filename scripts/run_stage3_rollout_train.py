import argparse
import json
import os
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from model.tokenizer import create_text_tokenizer, decode_text_token_ids, get_text_extra_tokens
from model.transformer import create_transformer, get_tokenizer_path, resolve_model_runtime_config
from scripts.run_stage2_tool_loop import checkpoint_needs_base_model, resolve_checkpoint_path
from utils.load import load_kg, load_model, load_yaml, resolve_sampled_dataset_path
from utils.rl_rewards import score_rollout_trajectory
from utils.tool_loop import extract_action_text, extract_dsl_text, run_action_tool_call


def iter_records(path, max_rows=0):
    rows = []
    with open(path, 'r', encoding='utf-8') as input_file:
        for row_index, line in enumerate(input_file, start=1):
            if max_rows > 0 and row_index > max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_policy_model(args, tokenizer, ntoken, device, model_runtime_config):
    checkpoint_path = resolve_checkpoint_path(args)
    if checkpoint_path:
        base_model = None
        if checkpoint_needs_base_model(checkpoint_path):
            special_tokens = {
                'PAD': tokenizer.pad_token_id,
                'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
                'END': tokenizer.eos_token_id,
            }
            base_model = create_transformer(
                ntoken=ntoken,
                special_tokens=special_tokens,
                model_name=args.modelname,
                vocab_size=ntoken,
                use_pretrained_weights=args.use_pretrained_text_model,
                model_runtime_config=model_runtime_config,
            )
        model, _, _, _, _ = load_model(
            checkpoint_path,
            'model',
            return_huggingface_model=True,
            epoch=args.resume_epoch,
            model=base_model,
        )
        model.to(device)
        return model

    special_tokens = {
        'PAD': tokenizer.pad_token_id,
        'START': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        'END': tokenizer.eos_token_id,
    }
    return create_transformer(
        ntoken=ntoken,
        special_tokens=special_tokens,
        model_name=args.modelname,
        vocab_size=ntoken,
        use_pretrained_weights=args.use_pretrained_text_model,
        model_runtime_config=model_runtime_config,
    ).to(device)


def build_prompt(observation_text, history):
    return '\n'.join([observation_text, *history]).strip()


@torch.no_grad()
def sample_completion(model, tokenizer, prompt, device, args):
    tokenized = tokenizer(prompt, return_tensors='pt').to(device)
    input_len = tokenized.input_ids.shape[-1]
    generation_kwargs = {
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask,
        'max_new_tokens': args.max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': True,
        'temperature': args.temperature,
    }
    if args.top_k > 0:
        generation_kwargs['top_k'] = args.top_k
    if args.top_p < 1.0:
        generation_kwargs['top_p'] = args.top_p
    output = model.generate(**generation_kwargs)
    generated_ids = output[0, input_len:].detach().cpu().tolist()
    completion = decode_text_token_ids(tokenizer, generated_ids, preserve_whitespace=True)
    return {
        'prompt': prompt,
        'generated_ids': generated_ids,
        'completion': completion,
    }


def rollout_once(model, tokenizer, kg, record, device, args):
    observation = str(record['observation_text']).strip()
    if not observation.startswith('OBS '):
        observation = 'OBS ' + observation

    history = []
    segments = []
    raw_generations = []

    for step in range(1, args.max_action_steps + 1):
        prompt = build_prompt(observation, history)
        segment = sample_completion(model, tokenizer, prompt, device, args)
        segments.append(segment)
        generated = segment['completion']
        raw_generations.append({'step': step, 'prompt': prompt, 'generated': generated})

        dsl_text = extract_dsl_text(generated)
        if dsl_text is not None:
            return {
                'observation': observation,
                'history': history,
                'dsl': dsl_text,
                'raw_generations': raw_generations,
                'segments': segments,
                'stopped_by': 'dsl',
            }

        action_text = extract_action_text(generated)
        if action_text is None:
            return {
                'observation': observation,
                'history': history,
                'dsl': '',
                'raw_generations': raw_generations,
                'segments': segments,
                'stopped_by': 'unparseable_generation',
            }

        try:
            tool_output = run_action_tool_call(
                observation_text=observation,
                action_text=action_text,
                kg=kg,
                graph_split=args.graph_split,
            )
        except Exception as exc:
            history.append(action_text)
            return {
                'observation': observation,
                'history': history,
                'dsl': '',
                'raw_generations': [*raw_generations, {'step': step, 'tool_error': str(exc)}],
                'segments': segments,
                'stopped_by': 'action_execution_error',
            }
        history.extend([action_text, tool_output['result_text']])

    prompt = build_prompt(observation, history)
    segment = sample_completion(model, tokenizer, prompt, device, args)
    segments.append(segment)
    generated = segment['completion']
    raw_generations.append({'step': 'final', 'prompt': prompt, 'generated': generated})
    return {
        'observation': observation,
        'history': history,
        'dsl': extract_dsl_text(generated) or generated.strip(),
        'raw_generations': raw_generations,
        'segments': segments,
        'stopped_by': 'max_action_steps',
    }


def segment_ce_loss(model, tokenizer, segment, device):
    generated_ids = [
        token_id
        for token_id in segment['generated_ids']
        if token_id != tokenizer.pad_token_id
    ]
    if not generated_ids:
        return None

    prompt_ids = tokenizer(segment['prompt'], return_tensors='pt').input_ids.to(device)
    generated = torch.tensor([generated_ids], dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, generated], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[-1]] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def rollout_policy_loss(model, tokenizer, rollout, advantage, device):
    losses = []
    for segment in rollout.get('segments', []):
        loss = segment_ce_loss(model, tokenizer, segment, device)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return None
    ce_loss = torch.stack(losses).mean()
    return ce_loss * float(advantage)


def save_rollout_checkpoint(args, model, optimizer, step, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(
        output_dir,
        f'{args.dataname}-{args.scale}-{args.max_answer_size}-rollout-{step}-text2text.pth',
    )
    torch.save(
        {
            'model': model,
            'optimizer': optimizer,
            'epoch': step,
            'loss_log': {'rollout_rl_step': step},
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_rollout_rl(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    kg = load_kg(args.dataname)
    config_model = load_yaml(args.config_model)
    model_runtime_config = resolve_model_runtime_config(args.modelname, config_model)
    tokenizer, ntoken = create_text_tokenizer(
        get_tokenizer_path(model_runtime_config),
        extra_tokens=get_text_extra_tokens(include_graph_tokens=False),
        closed_text_tokens=None,
        trust_remote_code=bool(model_runtime_config.get('trust_remote_code', False)),
    )
    model = load_policy_model(args, tokenizer, ntoken, device, model_runtime_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data_path = resolve_sampled_dataset_path(args.data_root, args.dataname, args.split)
    records = iter_records(data_path, max_rows=args.max_rows)
    if not records:
        raise RuntimeError(f'No records loaded from {data_path}')

    output_dir = args.output_dir or os.path.join(args.checkpoint_root, args.modelname)
    log_path = args.log_path or os.path.join(output_dir, f'rollout_rl_{args.scale}.jsonl')
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

    baseline = 0.0
    model.train()
    for step in range(1, args.max_steps + 1):
        record = random.choice(records)
        model.eval()
        rollout = rollout_once(model, tokenizer, kg, record, device, args)
        score = score_rollout_trajectory(
            rollout=rollout,
            target=record.get('logic_dsl', ''),
            observation_text=record['observation_text'],
            kg=kg,
            graph_samplers=kg.graph_samplers,
            graph_split=args.graph_split,
        )
        reward = float(score['stage3_reward'])
        advantage = max(min(reward - baseline, args.advantage_clip), -args.advantage_clip)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rollout_policy_loss(model, tokenizer, rollout, advantage, device)
        loss_value = 0.0
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        baseline = args.baseline_momentum * baseline + (1.0 - args.baseline_momentum) * reward
        row = {
            'step': step,
            'reward': reward,
            'baseline': baseline,
            'advantage': advantage,
            'loss': loss_value,
            'stopped_by': rollout.get('stopped_by', ''),
            'num_actions': score.get('num_actions', 0),
            'action_parse_rate': score.get('action_parse_rate', 0.0),
            'action_execution_rate': score.get('action_execution_rate', 0.0),
            'jaccard': score.get('jaccard', 0.0),
            'answer_recall': score.get('answer_recall', 0.0),
            'answer_precision': score.get('answer_precision', 0.0),
            'observation_text': record['observation_text'],
            'pred_dsl': rollout.get('dsl', ''),
        }
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
            print(json.dumps(row, ensure_ascii=False))

        if args.save_steps > 0 and step % args.save_steps == 0:
            checkpoint_path = save_rollout_checkpoint(args, model, optimizer, step, output_dir)
            print(f'# saved {checkpoint_path}')

    checkpoint_path = save_rollout_checkpoint(args, model, optimizer, args.max_steps, output_dir)
    print(f'# final checkpoint {checkpoint_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./sampled_data_abduction/')
    parser.add_argument('--split', default='train')
    parser.add_argument('-d', '--dataname', default='DBpedia50')
    parser.add_argument('--modelname', default='GPT2_6_act_nt')
    parser.add_argument('--config-model', default='configs/config-model.yml')
    parser.add_argument('--checkpoint-root', default='./ckpt/')
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--scale', default='default')
    parser.add_argument('--max-answer-size', type=int, default=8)
    parser.add_argument('--graph-split', default='train')
    parser.add_argument('--max-rows', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--max-action-steps', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--baseline-momentum', type=float, default=0.9)
    parser.add_argument('--advantage-clip', type=float, default=2.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--log-path', default='')
    parser.add_argument('--use-pretrained-text-model', action='store_true')
    parser.add_argument('--device', default='')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    train_rollout_rl(parse_args())


if __name__ == '__main__':
    main()
