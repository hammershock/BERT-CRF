# -*- coding: utf-8 -*-
# train.py
"""
基于BERT+CRF的命名实体识别(NER)任务
———————— A simple Practice
@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn

"""
import logging
import os
from typing import Dict, Iterator

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
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
    correct_tags_predictions = 0
    total_tags_predictions = 0
    all_tag_labels = []
    all_tag_preds = []
    all_cls_labels = []
    all_cls_preds = []

    for idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch)
        val_running_loss += loss.item()

        predictions, label_preds = model(batch["input_ids"], batch["attention_mask"])
        # 评估序列标注
        for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
            valid_labels = label[mask == 1]
            correct_tags_predictions += (torch.tensor(pred).to(device) == valid_labels).sum().item()
            total_tags_predictions += len(valid_labels)
            valid_labels = valid_labels.detach().cpu().numpy()
            all_tag_labels.extend(valid_labels)
            all_tag_preds.extend(pred)

        # 评估句子分类
        for pred, cls in zip(label_preds, batch["classes"]):
            pred_label = torch.argmax(pred, dim=-1).cpu().numpy()
            cls_label = cls.cpu().item()
            all_cls_labels.append(cls_label)
            all_cls_preds.append(pred_label)

        current_accuracy = correct_tags_predictions / total_tags_predictions
        sentence_classification_accuracy = accuracy_score(all_cls_labels, all_cls_preds)
        yield {"loss": val_running_loss / (idx + 1),
               "tags_acc": current_accuracy,
               "all_tag_labels": all_tag_labels,
               "all_tag_preds": all_tag_preds,
               "cls_acc": sentence_classification_accuracy,
               "all_cls_labels": all_cls_labels,
               "all_cls_preds": all_cls_preds}


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
    config = TrainerConfig.from_json_file("data/product_comments/train_config.json")
    config.print_config()
    data_config = DatasetConfig.from_json_file("data/product_comments/data.json")

    # ============== Model Metadata ==================
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)  # load the pretrained model

    plot_path = "./plots"

    model = BERT_CRF(config.bert_model_path,
                     num_labels=len(data_config.tags_map) if data_config.tags_map else 1,
                     num_classes=len(data_config.cls_map) if data_config.cls_map else 1,
                     ).to(config.device)

    if config.pretrained_model is not None:
        model.load_state_dict(torch.load(config.pretrained_model, map_location=config.device))
    train_set = make_ner_dataset(data_config.train_data, data_config, tokenizer, config.max_seq_len, overlap=config.overlap)
    val_set = make_ner_dataset(data_config.dev_data, data_config, tokenizer, config.max_seq_len, overlap=config.overlap)

    # prepare for training...
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=config.num_workers)
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
        plot_confusion_matrix(results["all_tag_labels"], results["all_tag_preds"], os.path.join(plot_path, f"tags_cm_{epoch}.png"), data_config.tags_map)
        plot_confusion_matrix(results["all_cls_labels"], results["all_cls_preds"], os.path.join(plot_path, f"cls_cm_{epoch}.png"), data_config.cls_map)
        if epoch % config.save_every == 0:
            ensure_dir_exists(config.save_path)
            torch.save(model.state_dict(), config.save_path)
