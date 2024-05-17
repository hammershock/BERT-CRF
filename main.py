from tqdm import tqdm
from transformers import BertTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
import logging  # Logging

from model import BERT_CRF
from datasets import NERDataset

from torch.optim import AdamW


if __name__ == '__main__':
    # ====== Training Hyperparameters ======
    num_epochs = 3
    batch_size = 2
    lr = 5e-5  # fine-tuning
    num_labels = 8
    # ============== Model Metadata ==================
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./bert-base-chinese")  # load the pretrained model

    model = BERT_CRF('bert-base-chinese', num_labels=num_labels)
    print(f'Number of layers: {model.bert.config.num_hidden_layers}')
    print(f'Vocabulary size: {model.bert.config.vocab_size}')
    print(f'Embedding dimension: {model.bert.config.hidden_size}')

    # Move model and tensors to CUDA (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = NERDataset('./data/train.txt', './data/train_TAG.txt', tokenizer)
    # train_dataset = NERDataset('./data/dev.txt', './data/dev_TAG.txt', tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # We MUST use the same label map with Train set!
    val_dataset = NERDataset('./data/dev.txt', './data/dev_TAG.txt', tokenizer, label_map=train_dataset.label_map)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)

    # TensorBoard and Logging Setup
    writer = SummaryWriter(log_dir='../../tf-logs/ner_experiment')  # default logging dir of auto-dl
    logging.basicConfig(filename='training.log', level=logging.INFO)

    val_accuracy = None
    val_loss = None

    print('Model loaded successfully')
    for epoch in range(num_epochs):  # Number of epochs can be adjusted
        running_loss = 0.0
        p_bar = tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        for idx, batch in p_bar:  # use tqdm to show the progress
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Save the model every 1/4 epoch
            if (idx + 1) % (len(train_dataloader) // 4) == 0:
                torch.save(model.state_dict(), f'./model_epoch_{epoch + 1}_batch_{idx + 1}.pth')
            p_bar.set_postfix(running_loss=running_loss / (idx + 1), val_loss=val_loss, val_acc=val_accuracy)

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss}")

        # Send the training loss after epoch to TensorBoard
        writer.add_scalar('Training Loss', epoch_loss, epoch)

        # Log the training loss
        logging.info(f'Epoch: {epoch + 1}, Training Loss: {epoch_loss}')

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        p_bar = tqdm(enumerate(val_dataloader), desc=f'Validation {epoch + 1}/{num_epochs}')
        with torch.no_grad():
            for idx, batch in p_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                loss = model(input_ids, attention_mask, labels=labels)
                val_running_loss += loss.item()

                predictions = model(input_ids, attention_mask)
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    valid_labels = label[mask == 1]
                    correct_predictions += (torch.tensor(pred).to(device) == valid_labels).sum().item()
                    total_predictions += len(valid_labels)
                p_bar.set_postfix(running_loss=running_loss / len(train_dataloader),
                                  val_loss=val_running_loss / (idx + 1),
                                  current_accuracy=correct_predictions / total_predictions)

        val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        # Send the validation loss and accuracy to TensorBoard
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        # Log the validation loss and accuracy
        logging.info(f'Epoch: {epoch + 1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

    writer.close()
