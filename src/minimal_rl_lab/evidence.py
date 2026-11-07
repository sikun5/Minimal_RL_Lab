"""Workspace-relative run configuration and training metrics."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time
from .serialization import JsonCodec

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_run_config(config, paths, out_dir, method, mode, ids, input_file, trainer_config):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dependencies = {}
    for name in ('torch', 'trl', 'transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    record = {
        'method': method,
        'mode': mode,
        'started_at': time.strftime('%F %T'),
        'command': f'rl-lab train {method} --mode {mode} --out {paths.record(out_dir)}',
        'input_file': paths.record(input_file),
        'input_sha256': sha256_file(input_file),
        'input_ids': list(ids),
        'config': config.to_dict(),
        'trainer': trainer_config.to_dict(),
        'dependencies': dependencies,
        'protocol': {'prompt': 'v2', 'scoring': 'answer_readout_r1'},
    }
    with open(Path(out_dir) / 'run_config.json', 'w', encoding='utf-8') as stream:
        JsonCodec(paths.root).dump(record, stream, indent=2, default=str)
    return record

def training_metrics(trainer, seconds, **extra):
    import torch
    return {
        'train_seconds': round(seconds, 1),
        'peak_cuda_allocated_GiB': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        'peak_cuda_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
        'peak_window': 'train() only; counters reset immediately before training',
        'trainer_global_step': getattr(trainer.state, 'global_step', None),
        'log_history': trainer.state.log_history,
        **extra,
    }

def finish_run(paths, out_dir, record, metrics):
    record['finished_at'] = time.strftime('%F %T')
    codec = JsonCodec(paths.root)
    for name, payload in (('run_config.json', record), ('metrics.json', metrics)):
        with open(Path(out_dir) / name, 'w', encoding='utf-8') as stream:
            codec.dump(payload, stream, indent=2, default=str)

def preparation_cost(paths):
    import json
    filename = paths.data / 'v2' / 'dpo_gen_stats.json'
    if not filename.exists():
        return None
    with filename.open(encoding='utf-8') as stream:
        stats = json.load(stream)
    return {
        'seconds': stats.get('elapsed_seconds'),
        'generated_tokens_including_eos': stats.get('total_generated_tokens_EOS_consistent'),
    }
