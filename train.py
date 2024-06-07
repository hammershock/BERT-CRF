# -*- coding: utf-8 -*-
# train.py
"""
基于BERT+CRF的命名实体识别(NER)任务
———————— A simple Practice
@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn

"""
import logging
from typing import Dict, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from config import TrainerConfig
from model import BERT_CRF
from ner_dataset import NERDataset
from utils import ensure_dir_exists, load_json_file


def train_epoch(model, train_dataloader, optimizer, device) -> Iterator[Dict[str, float]]:
    running_loss = 0.0
    model.train()

    for idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        yield {"running_loss": running_loss / (idx + 1)}


def validate(model, val_dataloader, optimizer, device) -> Iterator[Dict[str, float]]:
    model.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            val_running_loss += loss.item()

            predictions = model(batch["input_ids"], batch["attention_mask"])
            for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
                valid_labels = label[mask == 1]
                valid_preds = pred if isinstance(model, BERT_CRF) else pred[mask == 1]
                correct_predictions += (torch.tensor(valid_preds).to(device) == valid_labels).sum().item()
                total_predictions += len(valid_labels)

            current_accuracy = correct_predictions / total_predictions
            yield {"current_validation_loss": val_running_loss / (idx + 1), "current_accuracy": current_accuracy}


def tqdm_iteration(desc, model, dataloader, optimizer, device, train=True):
    func = train_epoch if train else validate
    p_bar = tqdm(func(model, dataloader, optimizer, device), desc=desc, total=len(train_loader))
    for results in p_bar:
        p_bar.set_postfix(**results)
    logging.info(f'Epoch: {epoch + 1} ' + " ".join(f"{k}: {v}" for k, v in results.items()))  # Log the training loss


if __name__ == '__main__':
    config = TrainerConfig.from_json_file("./config/train_config.json")
    ensure_dir_exists(config.save_path)
    ensure_dir_exists(config.log_path)

    # ============== Model Metadata ==================
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')  # load the pretrained model
    label_vocab: Dict[str, int] = load_json_file("./config/label_vocab.json")

    model = BERT_CRF('bert-base-chinese',
                     num_labels=len(label_vocab),
                     num_hidden_layers=12,
                     pretrained=2).to(config.device)

    train_set = NERDataset(config.max_seq_len, config.train_path, config.train_label_path, tokenizer, label_vocab,
                           special_label_id=label_vocab["O"])
    val_set = NERDataset(config.max_seq_len, config.val_path, config.val_label_path, tokenizer, label_vocab,
                         special_label_id=label_vocab["O"])

    # prepare for training...
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    optimizer = AdamW([
        {'params': list(model.bert.parameters()) + list(model.fc.parameters()), 'lr': config.lr},
        {'params': list(model.crf.parameters()), 'lr': config.lr_crf}
    ])

    logging.basicConfig(filename=config.log_path, level=logging.INFO)

    for epoch in range(config.num_epochs):
        # ============= Train ==============
        tqdm_iteration(f"Training {epoch + 1} / {config.num_epochs}", model, train_loader, optimizer, config.device)

        # ============= Validation ==============
        tqdm_iteration(f"Validation {epoch + 1} / {config.num_epochs}", model, val_loader, optimizer, config.device)

        if epoch % config.save_every == 0:
            torch.save(model.state_dict(), config.save_path)
