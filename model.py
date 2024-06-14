from enum import Enum, auto
from typing import NamedTuple, List, overload, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertModel, BertConfig


class InitializeMethod(Enum):
    random = auto()
    pretrained_embeddings = auto()
    pretrained_bert = auto()


def _init_weights(module, config):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def get_bert_model(bert_model_path, num_hidden_layers, cache_dir, pretrained) -> nn.Module:
    if pretrained in [InitializeMethod.random, InitializeMethod.pretrained_embeddings]:
        config = BertConfig.from_pretrained(bert_model_path, cache_dir=cache_dir)
        config.num_hidden_layers = num_hidden_layers
        # random Initialize
        bert = BertModel(config)
        if pretrained == InitializeMethod.pretrained_embeddings:  # use pretrained embeddings
            print(f"Initialized pretrained embeddings")
            pretrained_model = BertModel.from_pretrained(bert_model_path, cache_dir=cache_dir)
            bert.embeddings = pretrained_model.embeddings
            _init_weights(bert.encoder, bert.config)
    elif pretrained == InitializeMethod.pretrained_bert:
        print(f"Initialized pretrained_bert model")
        assert num_hidden_layers == 12
        bert = BertModel.from_pretrained(bert_model_path, cache_dir=cache_dir)
    else:
        raise ValueError("Unknown pretrained model Type")
    return bert


# labels/logits: [batch, seq_len]
Output = NamedTuple("Output", [("labels", List[List[int]]), ("emissions", torch.Tensor), ("cls_probs", torch.Tensor)])


class BertCRF(nn.Module):
    """bert-crf model for sequence labeling and sequence classification"""

    def __init__(self, bert_model_path, num_labels=1, num_classes=1, num_hidden_layers=12,
                 cache_dir='./bert-base-chinese', pretrained=InitializeMethod.pretrained_bert):
        super(BertCRF, self).__init__()
        self.bert = get_bert_model(bert_model_path, num_hidden_layers, cache_dir, pretrained)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.crf = CRF(num_labels, batch_first=True)

    @overload
    def forward(self, input_ids, attention_mask, label=None) -> Output:
        ...

    @overload
    def forward(self, input_ids, attention_mask, label: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, input_ids, attention_mask, tag_ids: Optional[torch.Tensor] = None,
                cls_ids: Optional[torch.Tensor] = None, **kwargs) -> Union[Output, torch.Tensor]:
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask)[:2]
        emissions = self.fc(self.dropout(sequence_output))
        cls_probs = self.classifier(self.dropout(pooled_output))

        if tag_ids is None and cls_ids is None:
            decoded_labels = self.crf.decode(emissions, mask=attention_mask.bool())  # decoded ids (integers)
            return Output(labels=decoded_labels, emissions=emissions, cls_probs=cls_probs)

        if tag_ids is not None and cls_ids is not None:
            crf_loss = -self.crf(emissions, tag_ids, mask=attention_mask.bool(), reduction='mean')
            cls_loss = F.cross_entropy(cls_probs, cls_ids)

            crf_weight = 1 / (crf_loss.item() ** 0.5 + 1e-8)
            cls_weight = 1 / (cls_loss.item() ** 0.5 + 1e-8)
            total_weight = crf_weight + cls_weight
            crf_loss = crf_loss * (crf_weight / total_weight)
            cls_loss = cls_loss * (cls_weight / total_weight)

            total_loss = crf_loss + cls_loss
        elif tag_ids is not None:
            total_loss = -self.crf(emissions, tag_ids, mask=attention_mask.bool(), reduction='mean')
        else:
            total_loss = self.loss_fct(cls_probs, cls_ids)

        return total_loss
