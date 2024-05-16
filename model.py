import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
import torch
from torch.utils.data import DataLoader

from ner_dataset import NERDataset

# FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version.
# Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
from torch.optim import AdamW


class BERT_CRF(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BERT_CRF, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir='./bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.fc(sequence_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte())
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())
            return prediction


if __name__ == '__main__':
    # ====== training HyperParameters ======

    num_epochs = 3
    batch_size = 16
    lr = 5e-5  # fine-tuning

    # ============== Model Metadata ==================
    # TODO: print the num_layers, num_tokens(vocabulary size), embed dim here...


    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # load the pretrained model
    # TODO: move the model and the tensors to cuda (if available).
    train_dataset = NERDataset('./data/train.txt', './data/train_TAG.txt', tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BERT_CRF('bert-base-chinese', num_labels=len(train_dataset.label_map))

    optimizer = AdamW(model.parameters(), lr=lr)
    print('model loaded successfully')
    model.train()
    for epoch in range(num_epochs):  # Number of epochs can be adjusted
        running_loss = 0.0
        p_bar = tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch}/{num_epochs}')
        for idx, batch in p_bar:  # use tqdm to show the progress
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # TODO: save the model every 1/4 epoch
            p_bar.set_postfix(running_loss=running_loss / (idx + 1))

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        # TODO: send the training loss after epoch to tensorboard,
        # TODO: And generate a log file using module `logging` (to record the training loss)

