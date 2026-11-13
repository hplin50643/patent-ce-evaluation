"""ModernBERT replacement with the author's pooler-shaped Quality scoring head.

ModernBERT supplies no pretrained pooler. The additional dense+tanh pooler uses
BERT-style normal(0, .02) / zero initialization; fc retains torch Linear defaults.
The full pretrained encoder is loaded separately and never reinitialized.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import ModernBertConfig, ModernBertModel
from transformers.models.modernbert.modeling_modernbert import ModernBertPreTrainedModel


class ModernContrastiveModel(ModernBertPreTrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        config.reference_compile = False
        super().__init__(config)
        self.bert = ModernBertModel(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.normal_(self.pooler.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pooler.bias)
        self.fc = nn.Linear(config.hidden_size, 1)
        # Do not call self.post_init(): the new head has explicit initializers,
        # while ModernBertModel has already initialized its own encoder.

    def compute_score(self, input_ids, attention_mask):
        encoded = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = torch.tanh(self.pooler(encoded.last_hidden_state[:, 0]))
        return torch.sigmoid(self.fc(pooled).squeeze(-1).float())

    def forward(self, input_ids, attention_mask):
        return self.compute_score(input_ids, attention_mask)

    def score_pair(self, sentence_a, sentence_b, tokenizer, device, max_length=2048):
        """Retain the public single-pair API; formal evaluation passes 4096."""
        inputs = tokenizer(sentence_a, sentence_b, return_tensors="pt",
                           padding="max_length", truncation=True,
                           max_length=max_length, return_token_type_ids=False)
        with torch.no_grad():
            score = self.compute_score(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
        return score.item()
