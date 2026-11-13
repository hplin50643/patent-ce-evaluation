from transformers import PreTrainedModel, LongformerConfig, LongformerModel
import torch.nn as nn
import torch

class ContrastiveModel(PreTrainedModel):
    
    config_class = LongformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = LongformerModel._from_config(config)
        self.fc = nn.Linear(config.hidden_size, 1)

    def compute_score(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  
        score = self.fc(pooled_output).squeeze()  
        score = torch.sigmoid(score)  
        return score
    
    def score_pair(self, sentence_a, sentence_b, tokenizer, device, max_length=2048):

        inputs = tokenizer(
            sentence_a, sentence_b,
            return_tensors="pt", padding="max_length", truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            score = self.compute_score(inputs["input_ids"], inputs["attention_mask"])
        return score.item()