from enum import Enum, auto
from typing import NamedTuple, List, overload

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
            pretrained_model = BertModel.from_pretrained(bert_model_path, cache_dir=cache_dir)
            bert.embeddings = pretrained_model.embeddings
            _init_weights(bert.encoder, bert.config)
    elif pretrained == InitializeMethod.pretrained_bert:
        assert num_hidden_layers == 12
        bert = BertModel.from_pretrained(bert_model_path, cache_dir=cache_dir)
    else:
        raise ValueError("Unknown pretrained model Type")
    return bert


# labels/logits: [batch, seq_len]
Output = NamedTuple("Output", [("labels", List[List[int]]), ("emissions", torch.Tensor)])


class BERT_CRF(nn.Module):
    """bert-crf model for sequence labeling and sequence classification"""

    def __init__(self, bert_model_path, num_labels=1, num_classes=1, num_hidden_layers=12,
                 cache_dir='./bert-base-chinese', pretrained=InitializeMethod.pretrained_bert):
        super(BERT_CRF, self).__init__()
        self.bert = get_bert_model(bert_model_path, num_hidden_layers, cache_dir, pretrained)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.crf = CRF(num_labels, batch_first=True)

    @overload
    def forward(self, input_ids, attention_mask, label: None = None) -> Output:
        ...

    @overload
    def forward(self, input_ids, attention_mask, label: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.fc(sequence_output)

        if labels is None:
            decoded_labels = self.crf.decode(emissions, mask=attention_mask.bool())  # decoded ids (integers)
            return Output(labels=decoded_labels, emissions=emissions)
        loss = -self.crf(emissions, labels, mask=attention_mask.bool())
        return loss


if __name__ == '__main__':
    model = BERT_CRF('bert-base-chinese', num_labels=9, pretrained=InitializeMethod.random)
    # model = BERT_Softmax('bert-base-chinese', num_labels=9, cache_dir="./bert-base-chinese")
    # print(model.bert.embeddings.word_embeddings.weight)  # 打印词嵌入参数
    print(f'Number of layers: {model.bert.config.num_hidden_layers}')
    print(f'Vocabulary size: {model.bert.config.vocab_size}')
    print(f'Embedding dimension: {model.bert.config.hidden_size}')
