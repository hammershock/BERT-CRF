import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig


class BERT_CRF(nn.Module):
    def __init__(self, bert_model_path, num_labels, num_hidden_layers=12, cache_dir='./bert-base-chinese', pretrained=0):
        super(BERT_CRF, self).__init__()
        if pretrained in [0, 1]:
            config = BertConfig.from_pretrained(bert_model_path, cache_dir=cache_dir)
            config.num_hidden_layers = num_hidden_layers
            # random Initialize
            self.bert = BertModel(config)
            if pretrained == 1:  # use pretrained embeddings
                pretrained_model = BertModel.from_pretrained(bert_model_path, cache_dir=cache_dir)
                self.bert.embeddings = pretrained_model.embeddings
                self._init_weights(self.bert.encoder)
        elif pretrained == 2:
            assert num_hidden_layers == 12
            self.bert = BertModel.from_pretrained(bert_model_path, cache_dir=cache_dir)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.fc(sequence_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction


if __name__ == '__main__':
    model = BERT_CRF('bert-base-chinese', num_labels=9, pretrained=False)
    # model = BERT_Softmax('bert-base-chinese', num_labels=9, cache_dir="./bert-base-chinese")
    # print(model.bert.embeddings.word_embeddings.weight)  # 打印词嵌入参数
    print(f'Number of layers: {model.bert.config.num_hidden_layers}')
    print(f'Vocabulary size: {model.bert.config.vocab_size}')
    print(f'Embedding dimension: {model.bert.config.hidden_size}')
