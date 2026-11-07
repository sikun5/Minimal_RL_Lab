"""Filter and split GSM8K using the configured tokenizer and seed."""
import os
import time
import random
import hashlib
from ..serialization import JsonCodec
from ..prompts import count_prompt_tokens_v2
from ..models import load_tokenizer

def run(config, paths, args):
    json = JsonCodec(paths.root)
    SPLITS_DIR = str(paths.data / 'splits_v2')
    SPLITS_V2 = SPLITS_DIR
    os.makedirs(str(paths.data), exist_ok=True)
    os.makedirs(SPLITS_V2, exist_ok=True)

    def log(message):
        print(f"[{time.strftime('%F %T')}] {message}", flush=True)
    import pandas as pd
    tok = load_tokenizer(config, paths)
    records = {}
    for split in ('train', 'test'):
        path = os.path.join(str(paths.data), 'hub', 'main', f'{split}-00000-of-00001.parquet')
        df = pd.read_parquet(path)
        rows = [{'id': f'{split}:{i}', 'question': str(r['question']), 'answer': str(r['answer'])} for i, r in df.iterrows()]
        records[split] = rows

    def build(rows, need_solution_tokens):
        kept, dp, ds = ([], 0, 0)
        for r in rows:
            ptoks = count_prompt_tokens_v2(tok, r['question'])
            if ptoks > config.generation.max_prompt_tokens:
                dp += 1
                continue
            if need_solution_tokens:
                stoks = len(tok.tokenize(r['answer']))
                if stoks > config.generation.max_completion_tokens:
                    ds += 1
                    continue
            kept.append({**r, 'prompt_tokens': ptoks})
        return (kept, dp, ds)
    train_pool, tp_drop, ts_drop = build(records['train'], True)
    test_pool, te_drop, _ = build(records['test'], False)
    log(f'v2 train pool: kept={len(train_pool)} drop_prompt={tp_drop} drop_solution={ts_drop}')
    log(f'v2 test pool: kept={len(test_pool)} drop_prompt={te_drop}')
    rng = random.Random(config.seed)
    st = train_pool[:]
    rng.shuffle(st)
    train, dev = (st[:config.dataset.train_size], st[config.dataset.train_size:config.dataset.train_size + config.dataset.dev_size])
    rng2 = random.Random(config.seed)
    sst = test_pool[:]
    rng2.shuffle(sst)
    test = sst[:config.dataset.test_size]
    v2_ids = {'train': [r['id'] for r in train], 'dev': [r['id'] for r in dev], 'test': [r['id'] for r in test]}
    assert len(train) == config.dataset.train_size and len(dev) == config.dataset.dev_size
    assert len(test) == config.dataset.test_size
    assert not set(v2_ids['train']) & set(v2_ids['dev'])
    for name, lst in (('train', train), ('dev', dev), ('test', test)):
        with open(os.path.join(SPLITS_V2, f'{name}.jsonl'), 'w') as f:
            for r in lst:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    id_fingerprints = {name: hashlib.sha256('\n'.join(ids).encode()).hexdigest() for name, ids in v2_ids.items()}
    summary = {
        'ordered_id_sha256': id_fingerprints,
        'protocol': 'v2',
        'model': {'repo_id': config.model.repo_id, 'revision': config.model.revision},
        'dataset': {'repo_id': config.dataset.repo_id, 'revision': config.dataset.revision},
        'counts': {
            'train_pool_kept': len(train_pool),
            'train_drop_prompt': tp_drop,
            'train_drop_solution': ts_drop,
            'test_pool_kept': len(test_pool),
            'test_drop_prompt': te_drop,
            'train': len(train),
            'dev': len(dev),
            'test': len(test),
        },
        'max_prompt_tokens_v2': {
            'train': max((r['prompt_tokens'] for r in train)),
            'dev': max((r['prompt_tokens'] for r in dev)),
            'test': max((r['prompt_tokens'] for r in test)),
        },
    }
    with open(os.path.join(str(paths.data), 'split_summary_v2.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log('DATA_PREPARATION_OK')
    return 0
