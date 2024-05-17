import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertTokenizer, BertConfig


class BERT_CRF(nn.Module):
    def __init__(self, bert_model_name, num_labels, num_hidden_layers=None, cache_dir='./bert-base-chinese',
                 pretrained=True):
        super(BERT_CRF, self).__init__()
        if pretrained:
            config = BertConfig.from_pretrained(bert_model_name, cache_dir=cache_dir)
            if num_hidden_layers is not None and config.num_hidden_layers != num_hidden_layers:
                # 发出警告
                print(
                    f"Warning: num_hidden_layers ({num_hidden_layers}) does not match the pre-trained model ({config.num_hidden_layers}).")

                # 设置新的层数
                config.num_hidden_layers = num_hidden_layers

                # 加载预训练模型以获取词嵌入
                pretrained_model = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
                self.bert = BertModel(config)

                # 迁移词嵌入层的权重
                self.bert.embeddings = pretrained_model.embeddings

                # 初始化其他层的参数
                self._init_weights(self.bert.encoder)
            else:
                self.bert = BertModel.from_pretrained(bert_model_name, config=config, cache_dir=cache_dir)
        else:
            config = BertConfig.from_pretrained(bert_model_name, cache_dir=cache_dir)
            if num_hidden_layers is not None:
                config.num_hidden_layers = num_hidden_layers
            self.bert = BertModel(config)

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


class BERT_Softmax(nn.Module):
    def __init__(self, bert_model_name, num_labels, cache_dir='./bert-base-chinese'):
        super(BERT_Softmax, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.fc(sequence_output)
        predictions = self.softmax(logits)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # CrossEntropyLoss expects inputs of shape (N, C) and targets of shape (N)
            # Flatten the inputs and targets
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)
            return loss
        else:
            return predictions


if __name__ == '__main__':
    model = BERT_CRF('bert-base-chinese', num_labels=9, pretrained=False)
    print(model.bert.embeddings.word_embeddings.weight)  # 打印词嵌入参数
    print(f'Number of layers: {model.bert.config.num_hidden_layers}')
    print(f'Vocabulary size: {model.bert.config.vocab_size}')
    print(f'Embedding dimension: {model.bert.config.hidden_size}')
