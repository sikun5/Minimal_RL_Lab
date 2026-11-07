# Minimal RL Lab

A minimal implementation of major RL post-training algorithms for LLMs. 
This repository compares **PPO**, **GRPO**, and **DPO** on **Qwen2.5-0.5B-Instruct** 
using a subset of **GSM8K**:

- **PPO & GRPO**: Supervised using rule-based numeric-correctness rewards.
- **DPO**: Trained on offline preference pairs with standard solutions as chosen answers.

## Repository layout

```text
configs/                Experiment configuration
requirements/           Pinned dependencies
src/minimal_rl_lab/     Data preparation, training, scoring and evaluation
```

Model weights and data are downloaded or generated with the commands below.

## Installation

Use **Linux, Python 3.10 and CUDA 12.1** with an NVIDIA GPU supporting BF16 and
bitsandbytes 4-bit loading. PPO must accommodate both the actor and critic.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/lock-cu121.txt
python -m pip install --no-deps -e .
```

The dependency lock uses TRL 0.15.2, Transformers 4.49.0, PEFT 0.14.0,
Accelerate 1.4.0, Datasets 3.4.1 and bitsandbytes 0.45.3. The PPO compatibility
module requires TRL 0.15.2.

## Workspace and configuration

Run the examples from the repository root. Runtime files default to
`./workspace`. Global arguments precede the command:

```bash
rl-lab --workspace ./workspace --config configs/gsm8k.json COMMAND
python -m minimal_rl_lab --workspace ./workspace --config configs/gsm8k.json COMMAND
```

Both entry points provide the same commands. Output, adapter and preference-file
arguments are relative to the workspace; `--config` is relative to the current
directory. Configuration files and run records use relative filesystem paths.

```text
workspace/
  models/Qwen2.5-0.5B-Instruct-7ae55760/
  cache/huggingface/
  data/hub/main/
  data/splits_v2/
  data/v2/
  runs/
```

Use the same workspace and configuration arguments for every stage.

## Reproduce the experiment

### 1. Download the model and dataset

```bash
rl-lab --config configs/gsm8k.json download
```

Fixed sources:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset: `openai/gsm8k`, configuration `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`.

If Hugging Face access requires a mirror, set the endpoint for the command:

```bash
HF_ENDPOINT=https://hf-mirror.com rl-lab --config configs/gsm8k.json download
```

The downloader uses the fixed revisions and at most three attempts per file.

### 2. Prepare the splits

```bash
rl-lab --config configs/gsm8k.json prepare-data
```

Filter the official training split by prompt and solution length, shuffle with
`random.Random(42)`, then take 512 training and 32 development questions. Filter
the official test split by prompt length, shuffle separately with seed 42, and
take 256 test questions. Prompt and solution limits are 512 tokens. The four-message
English prompt consists of a system instruction, a `2 + 3` format example, and
the question; the requested final answer format is `#### <number>`.

### 3. Prepare DPO preference pairs

```bash
rl-lab --config configs/gsm8k.json prepare-preferences
```

Use only training questions and the initial quantized model. With seed 42,
temperature 0.8, top-p 1 and batch size 16, make at most four attempts per question.
Choose the first R1-readable wrong answer as rejected and the complete standard
solution as chosen. Write accepted pairs and preparation statistics under
`workspace/data/v2`. Fewer than 128 accepted pairs returns a nonzero status.

### 4. Train independent adapters

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True rl-lab --config configs/gsm8k.json train ppo --mode full --out runs/ppo_full
rl-lab --config configs/gsm8k.json train grpo --mode full --out runs/grpo_full
rl-lab --config configs/gsm8k.json train dpo --mode full --out runs/dpo_full
```

Each method starts from the same untrained base. Training requires an explicit
`--mode full` or `--mode smoke`; default output directories follow
`runs/{method}_{mode}`. A run saves its configuration, dependency versions and
input IDs in `run_config.json`, training metrics in `metrics.json`, and the
adapter/tokenizer in `final/`.

### 5. Evaluate and summarize

```bash
rl-lab evaluate --out runs/eval_base
rl-lab evaluate --adapter runs/ppo_full/final --out runs/eval_ppo
rl-lab evaluate --adapter runs/grpo_full/final --out runs/eval_grpo
rl-lab evaluate --adapter runs/dpo_full/final --out runs/eval_dpo
rl-lab summarize --runs runs/eval_base runs/eval_ppo runs/eval_grpo runs/eval_dpo --out results/metrics.csv
```

Evaluation uses the same test questions, greedy decoding and a 512-token answer
budget. Each evaluation produces `predictions.jsonl` and `metrics.json`. The
summary recomputes R1/strict correctness and checks unique, identically ordered
question IDs. Evaluation's `new_tokens` counter measures batch-padded raw
generated length, which may include padding after EOS.

## Training configuration

All methods use 4-bit NF4 QLoRA, BF16 compute, gradient checkpointing, LoRA r=8,
alpha=16, dropout=0, `q_proj`/`v_proj`, seed 42 and one pass over their respective
training inputs. Prompt/answer budgets are 512/512 tokens.

| Method | Signal | LR | KL / beta | Microbatch / accumulation | Other settings |
|---|---|---:|---:|---:|---|
| PPO | One online answer and rule reward; LoRA critic/value head | 1e-5 | KL 0.02 | 1 / 8 | Clip 0.2; 2 PPO epochs per rollout batch; generation chunk 2 |
| GRPO | Four answers with group-relative rule rewards | 1e-5 | beta 0.02 | 4 / 2 | Record complete equal-reward groups |
| DPO | Standard chosen / readable wrong rejected | 1e-5 | beta 0.1 | 1 / 8 | One epoch over retained pairs |

The PPO value head is hidden-size → 1024 → 1 with ReLU. PPO uses eager attention
and float32 loading with BF16 quantized compute and scoped autocast. Its PPOConfig
explicitly sets response length 512 and temperature 0.8. The compatibility module
returns real per-chunk logits for TRL to consume and restores patched functions
on exit; it does not replace `selective_log_softmax` or change the loss.

## Scoring

R1 reads an explicitly marked final answer from `####`, `\boxed{...}` or
final-answer declarations. It compares parsed float values and rejects conflicting
values across supported markers. The declaration/boxed tail window is 400 characters;
ambiguity checks round values to six decimals. Strict accuracy uses the `####`
parser, which accepts a leading numeric token. Both metrics use the same
question set and record truncated responses separately.

## Original experiment results

These are archived experiment results run with a A800 MIG 1g.10gb GPU and 512 train / 32 dev / 256 test split on GSM8K.

| Model | # R1 correct | R1 accuracy | Strict accuracy | Pure training seconds |
|---|---:|---:|---:|---:|
| Base | 13 | 5.08% | 1.95% | — |
| PPO | 14 | 5.47% | 2.34% | 6941.9 |
| GRPO | 20 | 7.81% | 3.52% | 6648.1 |
| DPO | 12 | 4.69% | 2.73% | 155.2 |

Note that DPO additionally took 2009 seconds of preference generation.

