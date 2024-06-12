# -*- coding: utf-8 -*-
# train.py
"""
基于BERT+CRF的命名实体识别(NER)任务
———————— A simple Practice
@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn

"""
import logging
import os.path
from typing import Dict, Iterator

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from config import TrainerConfig, DatasetConfig
from model import BERT_CRF
from ner_dataset import make_ner_dataset
from utils import ensure_dir_exists


def train_epoch(model, data_loader, optimizer, device) -> Iterator[Dict[str, float]]:
    running_loss = 0.0
    model.train()

    for idx, batch in enumerate(data_loader):
        # (batch_size, seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        yield {"running_loss": running_loss / (idx + 1)}


@torch.no_grad()
def validate(model, data_loader, _, device) -> Iterator[Dict[str, float]]:
    """
    validate model on dev set
    :param model:
    :param data_loader:
    :param _: [unused] 为了让形式和`train_epoch`类似
    :param device:
    :return:
    """
    model.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    for idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch)
        val_running_loss += loss.item()

        predictions = model(batch["input_ids"], batch["attention_mask"])
        for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
            valid_labels = label[mask == 1]
            correct_predictions += (torch.tensor(pred).to(device) == valid_labels).sum().item()
            total_predictions += len(valid_labels)
            valid_labels = valid_labels.detach().cpu().numpy()
            all_labels.extend(valid_labels)
            all_predictions.extend(pred)

        current_accuracy = correct_predictions / total_predictions
        yield {"current_validation_loss": val_running_loss / (idx + 1),
               "current_accuracy": current_accuracy,
               "all_labels": all_labels,
               "all_predictions": all_predictions}


def plot_confusion_matrix(all_labels, all_preds, plot_path, label_map):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')

    # Plot the normalized confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(label_map.keys()))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    ensure_dir_exists(plot_path)
    plt.savefig(plot_path)
    plt.show()


def tqdm_iteration(desc, model, dataloader, optimizer, device, func):
    generator = func(model, dataloader, optimizer, device)
    p_bar = tqdm(generator, desc=desc, total=len(dataloader))
    for results in p_bar:
        p_bar.set_postfix(**{k: v for k, v in results.items() if isinstance(v, float)})
    return results


if __name__ == '__main__':
    config = TrainerConfig.from_json_file("./config/train_config.json")
    config.print_config()
    data_config = DatasetConfig.from_json_file("./config/data.json")

    # ============== Model Metadata ==================
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)  # load the pretrained model

    plot_path = "./plots"

    model = BERT_CRF(config.bert_model_path,
                     num_labels=len(data_config.tags_map) if data_config.tags_map else 1,
                     num_classes=len(data_config.cls_map) if data_config.cls_map else 1,
                     ).to(config.device)

    train_set = make_ner_dataset(data_config.train_data, data_config, tokenizer, config.max_seq_len, overlap=config.overlap)
    val_set = make_ner_dataset(data_config.dev_data, data_config, tokenizer, config.max_seq_len, overlap=config.overlap)

    # prepare for training...
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    optimizer = AdamW([
        {'params': list(model.bert.parameters()) + list(model.fc.parameters()), 'lr': config.lr},
        {'params': list(model.crf.parameters()), 'lr': config.lr_crf}
    ])

    ensure_dir_exists(config.log_path)
    logging.basicConfig(filename=config.log_path, level=logging.INFO)

    for epoch in range(config.num_epochs):
        # ============= Train ==============
        results = tqdm_iteration(f"Training {epoch + 1} / {config.num_epochs}", model, train_loader, optimizer, config.device, train_epoch)
        logging.info(f'Epoch: {epoch + 1} ' + " ".join(f"{k}: {v}" for k, v in results.items() if isinstance(v, float)))

        # ============= Validation ==============
        results = tqdm_iteration(f"Validation {epoch + 1} / {config.num_epochs}", model, val_loader, optimizer, config.device, validate)
        logging.info(f'Epoch: {epoch + 1} ' + " ".join(f"{k}: {v}" for k, v in results.items() if isinstance(v, float)))  # Log the training loss
        ensure_dir_exists(plot_path)
        plot_confusion_matrix(results["all_labels"], results["all_predictions"], os.path.join(plot_path, f"confusion_matrix_epoch{epoch}.png"), label_vocab)

        if epoch % config.save_every == 0:
            ensure_dir_exists(config.save_path)
            torch.save(model.state_dict(), config.save_path)
