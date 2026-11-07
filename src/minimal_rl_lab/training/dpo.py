"""DPO training on standard-solution/readable-wrong-answer preference pairs."""
import os
import time
import torch
from ..serialization import JsonCodec, sanitize_saved_metadata
from ..models import load_base, load_tokenizer, lora_config
from ..prompts import render_prompt_v2
from .. import evidence as fe

def run(config, paths, args):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    json = JsonCodec(paths.root)
    out_dir = args.out or str(paths.runs / f'dpo_{args.mode}')
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = load_tokenizer(config, paths)
    with open(args.pairs, encoding='utf-8') as stream:
        pairs = [json.loads(line) for line in stream]
    if args.mode == 'smoke':
        pairs = pairs[:8]
    dataset = Dataset.from_list(
        [{
            'prompt': render_prompt_v2(tokenizer, pair['question']),
            'chosen': pair['chosen'],
            'rejected': pair['rejected'],
            'id': pair['id'],
        } for pair in pairs],
    ).remove_columns(
        ['id'],
    )
    lora = lora_config(config)
    model = load_base(config, paths, ppo=False)
    dpo_cfg = DPOConfig(
        output_dir=out_dir,
        learning_rate=config.dpo.learning_rate,
        beta=config.dpo.beta,
        per_device_train_batch_size=config.dpo.microbatch,
        gradient_accumulation_steps=config.dpo.gradient_accumulation,
        num_train_epochs=1,
        max_prompt_length=config.generation.max_prompt_tokens,
        max_completion_length=config.generation.max_completion_tokens,
        seed=config.seed,
        bf16=True,
        gradient_checkpointing=config.adaptation.gradient_checkpointing,
        report_to=[],
        logging_steps=1 if args.mode == 'smoke' else 4,
        save_strategy='no',
        dataset_num_proc=1,
    )
    record = fe.write_run_config(
        config,
        paths,
        out_dir,
        'dpo',
        args.mode,
        [pair['id'] for pair in pairs],
        args.pairs,
        dpo_cfg,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train()
    metrics = fe.training_metrics(
        trainer,
        time.time() - started,
        n_pairs=len(pairs),
        preference_preparation=fe.preparation_cost(paths),
    )
    trainer.save_model(os.path.join(out_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'final'))
    fe.finish_run(paths, out_dir, record, metrics)
    sanitize_saved_metadata(os.path.join(out_dir, 'final'), paths.root)
    print('DPO_RUN_OK ->', out_dir)
    return 0
