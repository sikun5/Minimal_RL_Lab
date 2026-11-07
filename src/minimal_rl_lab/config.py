"""Experiment configuration and relative workspace paths."""
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
import json
import os

@dataclass
class ModelConfig:
    repo_id: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    revision: str = '7ae557604adf67be50417f59c2c2f167def9a775'

@dataclass
class DatasetConfig:
    repo_id: str = 'openai/gsm8k'
    name: str = 'main'
    revision: str = '740312add88f781978c0658806c59bc2815b9866'
    train_size: int = 512
    dev_size: int = 32
    test_size: int = 256

@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 1.0
    max_prompt_tokens: int = 512
    max_completion_tokens: int = 512
    eval_batch_size: int = 16
    preference_batch_size: int = 16

@dataclass
class AdaptationConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: ['q_proj', 'v_proj'])
    quant_type: str = 'nf4'
    gradient_checkpointing: bool = True

@dataclass
class PPOSettings:
    learning_rate: float = 1e-05
    kl_coef: float = 0.02
    cliprange: float = 0.2
    num_ppo_epochs: int = 2
    microbatch: int = 1
    gradient_accumulation: int = 8
    generation_chunk_size: int = 2

@dataclass
class GRPOSettings:
    learning_rate: float = 1e-05
    beta: float = 0.02
    num_generations: int = 4
    microbatch: int = 4
    gradient_accumulation: int = 2

@dataclass
class DPOSettings:
    learning_rate: float = 1e-05
    beta: float = 0.1
    microbatch: int = 1
    gradient_accumulation: int = 8
    max_samples_per_question: int = 4
    min_pairs: int = 128

@dataclass
class LabConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    ppo: PPOSettings = field(default_factory=PPOSettings)
    grpo: GRPOSettings = field(default_factory=GRPOSettings)
    dpo: DPOSettings = field(default_factory=DPOSettings)

    def to_dict(self):
        return asdict(self)

def load_config(filename=None):
    config = LabConfig()
    if filename is None:
        return config
    with open(filename, encoding='utf-8') as stream:
        overrides = json.load(stream)

    def update(obj, values):
        if not isinstance(values, dict):
            raise ValueError('Configuration sections must be JSON objects')
        known = {f.name for f in fields(obj)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        for key, value in values.items():
            current = getattr(obj, key)
            if is_dataclass(current):
                update(current, value)
            else:
                if isinstance(value, (dict, list)) and (not isinstance(current, type(value))):
                    raise ValueError(f'Invalid configuration type for {key}')
                setattr(obj, key, value)
    update(config, overrides)
    if config.generation.top_p != 1.0:
        raise ValueError('The locked PPO/GRPO generation path requires top_p=1.0')
    if config.dataset.name != 'main':
        raise ValueError('This experiment uses only the GSM8K main configuration')
    return config

@dataclass
class Workspace:
    """IO paths remain relative even when the caller supplies an absolute root."""
    root: Path
    model: Path
    data: Path
    runs: Path
    cache: Path

    @classmethod
    def create_paths(cls, workspace='./workspace', config=None):
        config = config or LabConfig()
        root = Path(os.path.relpath(os.fspath(workspace)))
        model = root / 'models' / ('Qwen2.5-0.5B-Instruct-' + config.model.revision[:8])
        return cls(root, model, root / 'data', root / 'runs', root / 'cache')

    def within(self, value):
        """CLI output/adapter arguments are workspace-relative; external roots are allowed."""
        p = Path(value)
        return Path(os.path.relpath(p)) if p.is_absolute() else self.root / p

    def record(self, value):
        return Path(os.path.relpath(os.fspath(value), self.root)).as_posix()
