from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, PreTrainedModel

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = 'model'

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor | None=None, attention_mask: torch.Tensor | None=None, **kwargs: Any) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True, **kwargs)
        token_rewards = self.regression_head(output.last_hidden_state).squeeze(-1)
        if attention_mask is None:
            ends = torch.full((token_rewards.shape[0], 1), token_rewards.shape[1] - 1, device=token_rewards.device, dtype=torch.long)
        else:
            ends = attention_mask.long().cumsum(dim=1).argmax(dim=1, keepdim=True)
        return token_rewards.gather(1, ends).squeeze(1)
