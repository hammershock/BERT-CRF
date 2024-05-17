import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertTokenizer


class BERT_CRF(nn.Module):
    def __init__(self, bert_model_name, num_labels, cache_dir='./bert-base-chinese', pretrained=True):
        super(BERT_CRF, self).__init__()
        if pretrained:
            self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        else:
            self.bert = BertModel(BertModel.config_class.from_pretrained(bert_model_name, cache_dir=cache_dir))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',
                                              cache_dir='./bert-base-chinese')  # load the pretrained model

    device = "cuda"
    model = BERT_CRF('bert-base-chinese', num_labels=len(tokenizer.get_vocab()))
    torch.save(model.state_dict(), "./test_model.pt")

