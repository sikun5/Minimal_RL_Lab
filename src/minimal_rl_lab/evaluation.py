"""Greedy evaluation with separate R1 and #### metrics."""
import os
import time
from .serialization import JsonCodec
from .models import load_tokenizer, load_base, quantization_config
from .prompts import render_prompt_v2
from .scoring import score_response_r1

def run(config, paths, args):
    json = JsonCodec(paths.root)
    import torch
    torch.manual_seed(config.seed)
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)
    split_file = os.path.join(args.splits_dir, f'{args.split}.jsonl')
    rows = [json.loads(l) for l in open(split_file)]
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    log(
        f"eval split={args.split} n={len(rows)} protocol={args.protocol} adapter={args.adapter or 'none'} limit={args.limit or 'all'}",
    )
    tok = load_tokenizer(config, paths)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    bnb = quantization_config(config)
    model = load_base(config, paths, ppo=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        log(f'loaded adapter from {args.adapter}')
    model.eval()
    render = render_prompt_v2
    prompts = [render(tok, r['question']) for r in rows]
    t0 = time.time()
    results = []
    BATCH = config.generation.eval_batch_size
    for bs in range(0, len(rows), BATCH):
        br = rows[bs:bs + BATCH]
        enc = tok(prompts[bs:bs + BATCH], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=config.generation.max_completion_tokens,
                do_sample=not True,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
            )
        n_in = enc['input_ids'].shape[1]
        texts = tok.batch_decode(out[:, n_in:], skip_special_tokens=True)
        eos_id = tok.eos_token_id
        for r, text, seq in zip(br, texts, out):
            new_ids = seq[n_in:]
            n_new = int(new_ids.shape[0])
            has_eos = bool((new_ids == eos_id).any())
            finish = 'eos' if has_eos else 'length_cap'
            sc = score_response_r1(text, r['answer'], truncated=finish == 'length_cap')
            results.append(
                {
                    'id': r['id'],
                    'question': r['question'],
                    'response': text,
                    'extracted': sc['extracted'],
                    'gold': sc['gold'],
                    'correct': sc['correct'],
                    'correct_strict': sc['correct_strict'],
                    'format_valid': sc['format_valid'],
                    'truncated': sc['truncated'],
                    'readout_source': sc['readout_source'],
                    'new_tokens': n_new,
                    'finish': finish,
                },
            )
        log(f'batch {bs // BATCH + 1}/{(len(rows) + BATCH - 1) // BATCH} done')
    elapsed = time.time() - t0
    n = len(results)
    n_correct = sum((r['correct'] for r in results))
    n_correct_strict = sum((r['correct_strict'] for r in results))
    n_valid = sum((1 for r in results if r['format_valid']))
    n_trunc = sum((1 for r in results if r['truncated']))
    gen_tokens = sum((r['new_tokens'] for r in results))
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 30
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
    pred_path = os.path.join(args.out, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    metrics = {
        'model': {'repo_id': config.model.repo_id, 'revision': config.model.revision, 'adapter': args.adapter},
        'prompt_protocol': args.protocol,
        'scoring_protocol': 'answer_readout_r1',
        'split': args.split,
        'n_questions': n,
        'n_correct': n_correct,
        'accuracy_r1': round(n_correct / n, 4) if n else None,
        'n_correct_strict': n_correct_strict,
        'accuracy_strict': round(n_correct_strict / n, 4) if n else None,
        'format_valid_rate_strict': round(n_valid / n, 4) if n else None,
        'truncated_rate': round(n_trunc / n, 4) if n else None,
        'total_generated_tokens': gen_tokens,
        'token_count_note': 'Raw batch-padded generated token count; may include padding after EOS',
        'elapsed_seconds': round(elapsed, 1),
        'peak_cuda_allocated_GB': round(peak_alloc, 3),
        'peak_cuda_reserved_GB': round(peak_reserved, 3),
        'decoding': 'greedy' if True else 'sampling',
        'mechanics_check_only': bool(args.limit and args.limit < 200),
    }
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(
        f'EVAL_DONE correct={n_correct}/{n} valid={n_valid} trunc={n_trunc} elapsed={elapsed:.0f}s peak={peak_alloc:.2f}GB',
    )
    print(json.dumps(metrics, indent=2))
    return 0
