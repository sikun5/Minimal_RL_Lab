"""Generate training-only DPO preference pairs."""
import os
import time
from ..serialization import JsonCodec
from ..prompts import render_prompt_v2, count_prompt_tokens_v2
from ..scoring import score_response_r1
from ..models import load_tokenizer, load_base, quantization_config

def run(config, paths, args):
    json = JsonCodec(paths.root)
    V2_DIR = str(paths.data / 'v2')
    SPLITS_DIR = str(paths.data / 'splits_v2')
    os.makedirs(str(paths.data), exist_ok=True)
    os.makedirs(V2_DIR, exist_ok=True)

    def log(message):
        print(f"[{time.strftime('%F %T')}] {message}", flush=True)
    import torch
    torch.manual_seed(config.seed)
    tok = load_tokenizer(config, paths)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    bnb = quantization_config(config)
    log('loading initial model (4-bit NF4, bf16)')
    model = load_base(config, paths, ppo=False)
    model.eval()
    train = [json.loads(l) for l in open(os.path.join(SPLITS_DIR, 'train.jsonl'))]
    log(f'train questions: {len(train)} (v2 splits)')
    prompts = [render_prompt_v2(tok, r['question']) for r in train]
    ptoks = [count_prompt_tokens_v2(tok, r['question']) for r in train]
    found = {r['id']: None for r in train}
    attempts = {r['id']: 0 for r in train}
    pairs = []
    samples_total = 0
    gen_tokens_total = 0
    t0 = time.time()
    BATCH = config.generation.preference_batch_size
    eos_id = tok.eos_token_id
    for pass_idx in range(1, config.dpo.max_samples_per_question + 1):
        active = [i for i, r in enumerate(train) if found[r['id']] is None]
        log(f'pass {pass_idx}: {len(active)} questions remaining')
        if not active:
            break
        for bs in range(0, len(active), BATCH):
            bidx = active[bs:bs + BATCH]
            enc = tok([prompts[i] for i in bidx], return_tensors='pt', padding=True).to('cuda')
            n_in = enc['input_ids'].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=config.generation.max_completion_tokens,
                    do_sample=True,
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            texts = tok.batch_decode(out[:, n_in:], skip_special_tokens=True)
            samples_total += len(bidx)
            for j, (i, text) in enumerate(zip(bidx, texts)):
                seq = out[j]
                new_ids = seq[n_in:]
                n_new_total = int(new_ids.shape[0])
                has_eos = bool((new_ids == eos_id).any())
                if has_eos:
                    eos_pos = (new_ids == eos_id).nonzero()[0].item()
                    completion_tokens = int(eos_pos) + 1
                    finish = 'eos'
                else:
                    completion_tokens = n_new_total
                    finish = 'length_cap'
                gen_tokens_total += completion_tokens
                rid = train[i]['id']
                attempts[rid] += 1
                sc = score_response_r1(text, train[i]['answer'], truncated=finish == 'length_cap')
                cat = 'valid_correct' if sc['correct'] == 1 else 'valid_wrong' if sc['extracted'] is not None else 'unreadable'
                if cat == 'valid_wrong' and found[rid] is None:
                    found[rid] = True
                    pairs.append(
                        {
                            'id': rid,
                            'question': train[i]['question'],
                            'chosen': train[i]['answer'],
                            'rejected': text,
                            'attempt': attempts[rid],
                            'prompt_protocol': 'v2',
                            'scoring_protocol': 'answer_readout_r1',
                            'prompt_tokens': ptoks[i],
                            'completion_tokens': completion_tokens,
                            'rejected_truncated': sc['truncated'],
                            'rejected_r1_value': sc['extracted'],
                            'gold': sc['gold'],
                            'readout_source': sc['readout_source'],
                        },
                    )
        log(f'pass {pass_idx} done; pairs so far: {len(pairs)}; elapsed={time.time() - t0:.0f}s')
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 30
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
    elapsed = time.time() - t0
    with open(os.path.join(V2_DIR, 'dpo_pairs.jsonl'), 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    hist = {}
    for r in train:
        a = attempts[r['id']]
        hist[a] = hist.get(a, 0) + 1
    skipped = sum((value is None for value in found.values()))
    stats = {
        'prompt_protocol': 'v2',
        'scoring_protocol': 'answer_readout_r1',
        'model': {
            'repo_id': config.model.repo_id,
            'revision': config.model.revision,
            'quantization': '4-bit NF4, bf16 compute',
        },
        'sampling': {
            'temperature': config.generation.temperature,
            'top_p': config.generation.top_p,
            'max_new_tokens': config.generation.max_completion_tokens,
            'max_samples_per_question': config.dpo.max_samples_per_question,
            'batch_size': BATCH,
            'seed': config.seed,
            'seed_semantics': 'torch.manual_seed(42) reset before full run',
        },
        'n_questions': len(train),
        'n_samples_generated': samples_total,
        'n_pairs': len(pairs),
        'attempts_histogram': {str(k): v for k, v in sorted(hist.items())},
        'skipped_after_max_attempts': skipped,
        'total_prompt_tokens': sum(ptoks),
        'total_generated_tokens_EOS_consistent': gen_tokens_total,
        'elapsed_seconds': round(elapsed, 1),
        'peak_cuda_allocated_GB': round(peak_alloc, 3),
        'peak_cuda_reserved_GB': round(peak_reserved, 3),
        'token_accounting': 'Completion tokens include the first EOS; batch padding after EOS is excluded.',
    }
    with open(os.path.join(V2_DIR, 'dpo_gen_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log(
        f'pairs={len(pairs)} samples={samples_total} elapsed={elapsed:.0f}s gen_tokens={gen_tokens_total} peak_alloc={peak_alloc:.2f}GB',
    )
    if len(pairs) < config.dpo.min_pairs:
        log(f'Insufficient pairs: {config.dpo.min_pairs} required')
        return 2
    log('DPO_V2_R1_OK')
    return 0
