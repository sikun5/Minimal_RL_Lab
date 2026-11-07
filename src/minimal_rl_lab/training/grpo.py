"""GRPO with four-answer groups and rule-based numeric rewards."""
import os
import time
import torch
from ..serialization import JsonCodec, sanitize_saved_metadata
from ..models import load_base, load_tokenizer, lora_config
from ..prompts import render_prompt_v2
from ..scoring import score_response_r1
from .. import evidence as fe

def run(config, paths, args):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    json = JsonCodec(paths.root)
    out_dir = args.out or str(paths.runs / f'grpo_{args.mode}')
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = load_tokenizer(config, paths)
    split_file = paths.data / 'splits_v2' / ('dev.jsonl' if args.mode == 'smoke' else 'train.jsonl')
    with split_file.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if args.mode == 'smoke':
        rows = rows[:8]
    dataset = Dataset.from_list(
        [{
            'prompt': render_prompt_v2(tokenizer, row['question']),
            'gold_text': row['answer'],
            'id': row['id'],
        } for row in rows],
    )
    per_question = {row['id']: [] for row in rows}

    def rule_reward_fn(prompts=None, completions=None, gold_text=None, id=None, **kwargs):
        rewards = []
        for pid, completion, gold in zip(id, completions, gold_text):
            score = score_response_r1(completion, gold, truncated=False)
            reward = float(score['correct'])
            rewards.append(reward)
            per_question[pid].append(reward)
        return rewards
    lora = lora_config(config)
    model = load_base(config, paths, ppo=False)
    model.enable_input_require_grads()
    grpo_cfg = GRPOConfig(
        output_dir=out_dir,
        learning_rate=config.grpo.learning_rate,
        beta=config.grpo.beta,
        num_generations=config.grpo.num_generations,
        per_device_train_batch_size=config.grpo.microbatch,
        gradient_accumulation_steps=config.grpo.gradient_accumulation,
        num_train_epochs=1,
        max_prompt_length=config.generation.max_prompt_tokens,
        max_completion_length=config.generation.max_completion_tokens,
        temperature=config.generation.temperature,
        seed=config.seed,
        bf16=True,
        gradient_checkpointing=config.adaptation.gradient_checkpointing,
        report_to=[],
        logging_steps=1,
        save_strategy='no',
    )
    record = fe.write_run_config(
        config,
        paths,
        out_dir,
        'grpo',
        args.mode,
        [row['id'] for row in rows],
        split_file,
        grpo_cfg,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[rule_reward_fn],
        args=grpo_cfg,
        train_dataset=dataset,
        peft_config=lora,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train()
    metrics = fe.training_metrics(trainer, time.time() - started, n_questions=len(rows))
    complete = [rewards for rewards in per_question.values() if len(rewards) == config.grpo.num_generations]
    zero_variance = sum((len(set(rewards)) <= 1 for rewards in complete))
    all_rewards = [reward for rewards in per_question.values() for reward in rewards]
    metrics.update(
        n_completions=len(all_rewards),
        n_complete_groups=len(complete),
        zero_variance_groups=zero_variance,
        zero_variance_group_rate=zero_variance / len(complete) if complete else None,
        mean_reward=sum(all_rewards) / len(all_rewards) if all_rewards else None,
    )
    trainer.save_model(os.path.join(out_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'final'))
    fe.finish_run(paths, out_dir, record, metrics)
    sanitize_saved_metadata(os.path.join(out_dir, 'final'), paths.root)
    print('GRPO_RUN_OK ->', out_dir)
    return 0
