"""Download the configured immutable model and dataset revisions."""
import os
import time
from ..serialization import JsonCodec
from ..evidence import sha256_file

def run(config, paths, args):
    json = JsonCodec(paths.root)
    os.makedirs(str(paths.data), exist_ok=True)

    def log(message):
        print(f"[{time.strftime('%F %T')}] {message}", flush=True)
    from huggingface_hub import snapshot_download, hf_hub_download
    log(f'download model {config.model.repo_id}@{config.model.revision} -> {str(paths.model)}')
    model_path = None
    for attempt in range(1, 4):
        try:
            model_path = snapshot_download(
                repo_id=config.model.repo_id,
                revision=config.model.revision,
                local_dir=str(paths.model),
                cache_dir=str(paths.cache / 'huggingface'),
                allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*', 'merges*', 'vocab*'],
                ignore_patterns=['original/*', '*.pth', 'consolidated*', '*arrow*'],
            )
            break
        except Exception as e:
            log(f'model attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}')
            time.sleep(10)
    if model_path is None:
        log('MODEL_DOWNLOAD_FAILED')
        return 1
    log('model download done')
    parquets = {}
    for split in ('train', 'test'):
        fn = f'main/{split}-00000-of-00001.parquet'
        lp = None
        for attempt in range(1, 4):
            try:
                lp = hf_hub_download(
                    repo_id=config.dataset.repo_id,
                    repo_type='dataset',
                    revision=config.dataset.revision,
                    filename=fn,
                    local_dir=os.path.join(str(paths.data), 'hub'),
                    cache_dir=str(paths.cache / 'huggingface'),
                )
                break
            except Exception as e:
                log(f'{fn} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}')
                time.sleep(10)
        if lp is None:
            log('DATASET_DOWNLOAD_FAILED')
            return 1
        parquets[split] = lp
        log(f'dataset {fn} -> {lp}')
    import pandas as pd
    from transformers import AutoTokenizer
    counts = {}
    for split, lp in parquets.items():
        df = pd.read_parquet(lp)
        counts[split] = len(df)
        log(f'parquet {split}: rows={len(df)} cols={list(df.columns)}')
    assert counts['train'] == 7473, f"unexpected train rows {counts['train']}"
    assert counts['test'] == 1319, f"unexpected test rows {counts['test']}"
    tok = AutoTokenizer.from_pretrained(str(paths.model), revision=config.model.revision)
    msgs = [{'role': 'user', 'content': 'hello'}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert '<|im_start|>' in rendered, 'chat template not Qwen-style'
    log(f'tokenizer ok; template sample: {rendered[:60]!r}')
    files = []
    for root, _, fns in os.walk(str(paths.model)):
        for fn in sorted(fns):
            fp = os.path.join(root, fn)
            entry = {'path': os.path.relpath(fp, str(paths.model)), 'size': os.path.getsize(fp)}
            if entry['size'] < 50 * 1024 * 1024:
                entry['sha256'] = sha256_file(fp)
            files.append(entry)
    files.sort(key=lambda e: e['path'])
    total_size = sum((e['size'] for e in files))
    manifest = {
        'model': {
            'repo_id': config.model.repo_id,
            'revision': config.model.revision,
            'endpoint': os.environ.get('HF_ENDPOINT'),
            'local_dir': str(paths.model),
            'n_files': len(files),
            'total_bytes': total_size,
            'files': files,
        },
        'dataset': {
            'repo_id': config.dataset.repo_id,
            'config': 'main',
            'revision': config.dataset.revision,
            'endpoint': os.environ.get('HF_ENDPOINT'),
            'files': {s: {'path': p, 'rows': counts[s]} for s, p in parquets.items()},
        },
        'verification': {'parquet_rows': counts, 'tokenizer_chat_template_ok': True},
    }
    out = os.path.join(str(paths.data), 'manifest_download.json')
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f'manifest written: {out}')
    log('DOWNLOAD_ALL_OK')
    return 0
