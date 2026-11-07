"""Scoped BF16 forward and chunked generation for TRL 0.15.2 PPO."""
from .logits import FreeingLogits

class PPOPatch:

    def __init__(self, bindings):
        self.bindings = bindings
        self.generation_config = None
        self.last_logits = None

    def restore(self):
        for module, attribute, function in self.bindings:
            setattr(module, attribute, function)

def install_ppo_compat():
    import torch
    from importlib.metadata import version
    if version('trl') != '0.15.2':
        raise RuntimeError('PPO compatibility requires trl==0.15.2')
    import trl.trainer.utils as tutils
    import trl.trainer.ppo_trainer as ppt
    if ppt.selective_log_softmax is not tutils.selective_log_softmax:
        raise RuntimeError('Unexpected selective_log_softmax override')
    patch = PPOPatch(
        [(tutils, 'batch_generation', tutils.batch_generation), (ppt, 'batch_generation', ppt.batch_generation), (tutils, 'forward', tutils.forward), (ppt, 'forward', ppt.forward), (ppt, 'get_reward', ppt.get_reward)],
    )
    generate = tutils.generate
    forward = tutils.forward
    get_reward = tutils.get_reward

    def chunked_generation(model, queries, local_rollout_forward_batch_size, pad_token_id, generation_config):
        patch.generation_config = {key: getattr(generation_config, key, None) for key in ('max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 'eos_token_id', 'pad_token_id')}
        size = local_rollout_forward_batch_size
        query_responses, logits = ([], [])
        for index in range(0, queries.shape[0], size):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                response, values = generate(model, queries[index:index + size], pad_token_id, generation_config)
            query_responses.append(response)
            logits.append(values)
        result = torch.cat(query_responses, 0)
        patch.last_logits = FreeingLogits(logits, size)
        return (result, patch.last_logits)

    def bf16_forward(*args, **kwargs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return forward(*args, **kwargs)

    def reward_with_context(model, query_responses, pad_token_id, context_length):
        if hasattr(model, 'holder'):
            model.holder['context_length'] = context_length
        return get_reward(model, query_responses, pad_token_id, context_length)
    tutils.batch_generation = chunked_generation
    ppt.batch_generation = chunked_generation
    tutils.forward = bf16_forward
    ppt.forward = bf16_forward
    ppt.get_reward = reward_with_context
    return patch
