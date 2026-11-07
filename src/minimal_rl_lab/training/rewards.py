"""PPO's rule reward adapter for the locked TRL get_reward contract."""
import torch
from ..scoring import score_response_r1

class _IdBackbone(torch.nn.Module):
    """Minimal backbone: stashes input_ids for the reward computation and
    returns a dummy output with hidden_states (get_reward contract)."""

    def __init__(self, holder):
        super().__init__()
        self.holder = holder
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, return_dict=True, output_hidden_states=False, use_cache=False, **kw):
        self.holder['input_ids'] = input_ids
        hidden = self.dummy.detach().expand(input_ids.shape[0], input_ids.shape[1], 1)

        class _Out:
            pass
        o = _Out()
        o.hidden_states = (hidden,)
        return o

class RuleRewardModel(torch.nn.Module):
    """Terminal rule reward (R1 correctness in {0,1}) under the get_reward contract."""
    base_model_prefix = 'backbone'

    def __init__(self, tokenizer, prompt2gold):
        super().__init__()
        self.tok = tokenizer
        self.prompt2gold = prompt2gold
        self.holder = {}
        self.backbone = _IdBackbone(self.holder)

    def _compute(self, input_ids, context_length):
        rewards = []
        pad_id = self.tok.pad_token_id
        for row in input_ids:
            ids = row.tolist()
            resp_ids = ids[context_length:]
            prompt_text = self.tok.decode(
                [t for t in ids[:context_length] if t != 0 and t != pad_id],
                skip_special_tokens=True,
            )
            resp_text = self.tok.decode(resp_ids, skip_special_tokens=True)
            gold = self.prompt2gold.get(prompt_text.replace('!', '')[-100:])
            if gold is None:
                rewards.append(0.0)
                continue
            sc = score_response_r1(resp_text, gold, truncated=False)
            rewards.append(float(sc['correct']))
        return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)

    def score(self, hidden_states):
        input_ids = self.holder.get('input_ids')
        ctx = self.holder.get('context_length')
        r = self._compute(input_ids, ctx)
        b, seq = (hidden_states.shape[0], hidden_states.shape[1])
        per_token = torch.zeros(b, seq, 1, device=hidden_states.device, dtype=torch.float32)
        per_token[:, -1, 0] = r
        return per_token
