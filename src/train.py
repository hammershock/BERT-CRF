# -*- coding: utf-8 -*-
# train.py
"""
基于BERT+CRF的命名实体识别(NER)任务
———————— A simple Practice
@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn

"""

import argparse
import os

from tqdm import tqdm
from transformers import BertTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
import logging  # Logging

from model import BERT_CRF, BERT_Softmax
from datasets import NERDataset

from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT-CRF model for NER.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_len", type=int, default=300, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=150, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--lr_crf", type=float, default=5e-3, help="Learning rate of CRF")
    parser.add_argument("--num_labels", type=int, default=9, help="Number of labels")
    parser.add_argument("--pretrained", type=int, default=2, help="Use pre-trained model embeddings, 0 for RandomInit, 1 for pretrained embeddings, 2 for bert pretrained weights")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers in BERT")
    parser.add_argument("--save_dir", type=str, default="../models", help="Directory to save models")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every N epochs")
    parser.add_argument("--log_path", type=str, default=f"../logs/train.log", help="Directory to save logs")
    parser.add_argument("--model_type", type=str, default="bert_crf", help="Save model every N epochs")
    return parser.parse_args()


def train_epoch(model, train_dataloader, optimizer, device, epoch, num_epochs, save_dir, save_every=1):
    running_loss = 0.0
    p_bar = tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', total=len(train_dataloader))
    model.train()
    for idx, batch in p_bar:  # use tqdm to show the progress
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Save the model every 1/4 epoch
        p_bar.set_postfix(running_loss=running_loss / (idx + 1))
    if epoch % save_every == 0:
        save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)

    epoch_loss = running_loss / len(train_dataloader)
    return epoch_loss


def validate(model, val_dataloader, device, epoch, num_epochs):
    # Validation loop
    model.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    p_bar = tqdm(enumerate(val_dataloader), desc=f'Validation {epoch + 1}/{num_epochs}', total=len(val_dataloader))
    with torch.no_grad():
        for idx, batch in p_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            val_running_loss += loss.item()

            predictions = model(batch["input_ids"], batch["attention_mask"])
            for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
                valid_labels = label[mask == 1]
                valid_preds = pred if isinstance(model, BERT_CRF) else pred[mask == 1]
                correct_predictions += (torch.tensor(valid_preds).to(device) == valid_labels).sum().item()
                total_predictions += len(valid_labels)
            p_bar.set_postfix(val_loss=val_running_loss / (idx + 1),
                              current_accuracy=correct_predictions / total_predictions)

    val_loss = val_running_loss / len(val_dataloader)
    val_accuracy = correct_predictions / total_predictions
    return val_loss, val_accuracy


if __name__ == '__main__':
    args = parse_args()

    # ====== Training Hyperparameters ======
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    num_labels = args.num_labels
    num_hidden_layers = args.num_hidden_layers
    save_dir = args.save_dir
    save_every = args.save_every
    max_length = args.max_len
    pretrained = args.pretrained
    log_filename = args.log_path
    lr_crf = args.lr_crf

    # ============== Model Metadata ==================
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./bert-base-chinese")  # load the pretrained model
    log_dir = os.path.split(log_filename)[0]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_type == "bert_crf":
        model = BERT_CRF('bert-base-chinese', num_labels=num_labels, num_hidden_layers=num_hidden_layers, pretrained=pretrained)
    elif args.model_type == "bert_softmax":
        model = BERT_Softmax('bert-base-chinese', num_labels=num_labels, cache_dir="./bert-base-chinese")
    else:
        raise ValueError("Invalid Model type, Model type must be either 'bert_crf' or 'bert_softmax'")

    print(f'Number of layers: {model.bert.config.num_hidden_layers}')
    print(f'Vocabulary size: {model.bert.config.vocab_size}')
    print(f'Embedding dimension: {model.bert.config.hidden_size}')

    model = BERT_CRF('bert-base-chinese', num_labels=num_labels, num_hidden_layers=num_hidden_layers, pretrained=pretrained)
    # Move model and tensors to CUDA (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    save_dir = f'./models/type{args.model_type}_layers_{num_hidden_layers}_pretrained{pretrained}'
    os.makedirs(save_dir, exist_ok=True)

    label_map = {'I_T': 8, 'I_PER': 7, 'B_LOC': 2, 'B_PER': 0, 'B_T': 3, 'B_ORG': 6, 'I_LOC': 4, 'O': 5, 'I_ORG': 1}
    train_set = NERDataset('./data/train.txt', './data/train_TAG.txt', tokenizer, max_len=max_length, label_map=label_map)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=18)
    val_set = NERDataset('./data/dev.txt', './data/dev_TAG.txt', tokenizer, label_map=label_map, max_len=512)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=18)    # We MUST use the same label map with Train set!

    optimizer = AdamW([
        {'params': list(model.bert.parameters()) + list(model.fc.parameters()), 'lr': lr},
        {'params': list(model.crf.parameters()), 'lr': lr_crf}
    ]) if args.model_type == "bert_crf" else AdamW(model.parameters(), lr=lr)

    # TensorBoard and Logging Setup
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    print('Model loaded successfully')
    for epoch in range(num_epochs):  # Number of epochs can be adjusted
        epoch_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs, save_dir, save_every)

        # Record Training Results
        logging.info(f'Epoch: {epoch + 1}, Training Loss: {epoch_loss}')  # Log the training loss

        val_loss, val_accuracy = validate(model, val_loader, device, epoch, num_epochs)

        # Record Validation Results
        logging.info(f'Epoch: {epoch + 1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')