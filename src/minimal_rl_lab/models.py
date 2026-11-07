"""Lazy model-loading helpers shared by data preparation, training and evaluation."""

def load_tokenizer(config, paths, padding_side='left'):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(paths.model), revision=config.model.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer

def quantization_config(config):
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.adaptation.quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def lora_config(config):
    from peft import LoraConfig
    a = config.adaptation
    return LoraConfig(
        r=a.r,
        lora_alpha=a.alpha,
        lora_dropout=a.dropout,
        target_modules=a.target_modules,
        task_type='CAUSAL_LM',
    )

def load_base(config, paths, *, ppo=False):
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = {'torch_dtype': torch.float32 if ppo else torch.bfloat16, 'device_map': {'': 0}}
    if ppo:
        kwargs['attn_implementation'] = 'eager'
    return AutoModelForCausalLM.from_pretrained(
        str(paths.model),
        revision=config.model.revision,
        quantization_config=quantization_config(config),
        **kwargs,
    )
