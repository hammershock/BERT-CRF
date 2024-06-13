# -*- coding: utf-8 -*-
# train.py
"""
基于BERT+CRF的命名实体识别(NER)任务
———————— A simple Practice
@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn

"""
import argparse
import csv
import logging
import os
import warnings
from collections import defaultdict
from functools import wraps
from typing import Dict, Iterator, Callable, Any

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from config import TrainerConfig, DatasetConfig
from model import BERT_CRF
from ner_dataset import make_dataset_from_config
from fgm_attack import FGM
from plot_utils import plot_confusion_matrix, plot_auc_curve
from utils import ensure_dir_exists


def tqdm_iteration(desc: str, func: Callable[..., Iterator[Dict[str, Any]]]):
    """iteration helper"""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        generator = func(*args, **kwargs)
        assert isinstance(args[1], DataLoader), "args[1] must be DataLoader"
        if 'epoch' not in kwargs:
            warnings.warn("attr epoch not found in kwargs, not able to display the current iteration progress")
        progress = f" epoch {kwargs.get('epoch', 0) + 1}/{len(args[1])}"
        p_bar = tqdm(generator, desc=desc + progress, total=len(args[1]))  # args[1] should be dataloader
        result = None
        for results in p_bar:
            p_bar.set_postfix(**{k: v for k, v in results.items() if isinstance(v, float)})
            result = results
        return result

    return wrapper


def with_tqdm(desc: str):
    def decorator(func: Callable[..., Iterator[Dict[str, Any]]]):
        return tqdm_iteration(desc, func)

    return decorator


@with_tqdm("Training")
def train_epoch(model, data_loader, optimizer, device, *, fgm=None, epoch=None) -> Dict[str, Any]:
    running_loss = 0.0
    model.train()

    for idx, batch in enumerate(data_loader):
        # (batch_size, seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        # the model will automatically handle the multiple inputs, and compute loss
        loss = model(**batch)  # when labels provided, it returns loss
        loss.backward()

        if fgm is not None:
            fgm.attack()
            loss_adv = model(**batch)
            loss_adv.backward()
            fgm.restore()

        optimizer.step()
        running_loss += loss.item()
        yield {"running_loss": running_loss / (idx + 1)}


@with_tqdm("Validating")
@torch.no_grad()
def validate(model, data_loader, device, *, epoch=None) -> Dict[str, Any]:
    """
    validate model on dev set
    :param model:
    :param data_loader:
    :param device:
    :param epoch: current epoch progress
    :return:
    """
    model.eval()
    ret = defaultdict(list)

    for idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions, label_logits = model(batch["input_ids"], batch["attention_mask"])
        # predictions: [batch, seq_len]
        # label_logits: Tensor[batch, num_cls]
        # TODO: is crf output a Tensor? Probably not!
        if "labels" in batch:  # 评估序列标注
            for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
                # pred: [seq_len]
                valid_labels = label[mask == 1].detach().cpu().numpy()
                ret['tag_gts'].extend(valid_labels)
                ret['tag_preds'].extend(pred)

        if "classes" in batch:  # 评估句子分类
            cls_probs = F.softmax(label_logits, dim=-1)
            cls_labels = torch.argmax(label_logits, dim=-1)  # [batch]
            ret['cls_preds'].extend(cls_labels.cpu().numpy())
            ret['cls_gts'].extend(batch['classes'].cpu().numpy())
            ret['cls_probs'].extend(cls_probs.cpu().numpy())
        yield ret


@torch.no_grad()
def test(model, data_loader, device, output_path):
    """
    Test the model and write the results to a CSV file
    :param model:
    :param data_loader:
    :param device:
    :param output_path: Path to the output CSV file
    """
    model.eval()
    results = []

    for idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions, label_logits = model(batch["input_ids"], batch["attention_mask"])
        label_preds = F.softmax(label_logits, dim=-1)

        for i, (pred, label) in enumerate(zip(predictions, batch["labels"])):
            bio_anno = " ".join(pred)
            cls_label = torch.argmax(label_preds[i], dim=-1).item()
            results.append([idx * len(predictions) + i, bio_anno, cls_label])

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "BIO_anno", "class"])
        writer.writerows(results)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Dataset configuration")
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--data_config', type=str, required=True)
    return parser.parse_args()


def plot_val_results(val_results: Dict[str, Any], epoch: int, plot_path, tags_map, cls_map) -> None:
    ensure_dir_exists(plot_path)
    tag_cm_path = os.path.join(plot_path, f"tags_cm_{epoch}.png")
    cls_cm_path = os.path.join(plot_path, f"cls_cm_{epoch}.png")
    auc_curve_path = os.path.join(plot_path, f"auc_curve_{epoch}.png")

    plot_confusion_matrix(val_results.get("tag_gts"), val_results.get("tag_preds"), tag_cm_path, tags_map)
    plot_confusion_matrix(val_results.get("cls_gts"), val_results.get("cls_preds"), cls_cm_path, cls_map)
    plot_auc_curve(val_results.get("cls_gts"), val_results.get("cls_probs"), auc_curve_path, cls_map)


if __name__ == '__main__':
    args = parse_arguments()
    # load configs
    config = TrainerConfig.from_json_file(args.train_config)
    config.print_config()
    data_config = DatasetConfig.from_json_file(args.data_config)

    """
    Text labeling and Text classification, two tasks all in one model!
    You can train the model using either kind of data
    """
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)

    # define model
    model = BERT_CRF(config.bert_model_path,
                     num_labels=len(data_config.tags_map) if data_config.tags_map else 1,
                     num_classes=len(data_config.cls_map) if data_config.cls_map else 1,
                     ).to(config.device)

    # load model from checkpoint
    if config.pretrained_model is not None:
        model.load_state_dict(torch.load(config.pretrained_model, map_location=config.device))

    # datasets
    train_set = make_dataset_from_config(data_config.train_data, data_config, tokenizer)
    val_set = make_dataset_from_config(data_config.dev_data, data_config, tokenizer)

    # prepare for training here
    # dataloaders, optimizer and logger
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    optimizer = AdamW([
        {'params': list(model.bert.parameters()) + list(model.fc.parameters()), 'lr': config.lr},
        {'params': list(model.crf.parameters()), 'lr': config.lr_crf}
    ])

    ensure_dir_exists(config.log_path)
    logging.basicConfig(filename=config.log_path, level=logging.INFO)

    fgm = FGM(model, epsilon=1.0)  # fgm attacker for embedding layers

    for epoch in range(config.num_epochs):
        # train the model with batched data, how to train depends on the data keys
        train_results = train_epoch(model, train_loader, optimizer, device=config.device, fgm=None, epoch=epoch)
        logging.info(f"Epoch {epoch}, running loss: {train_results['running_loss']}")
        # model validation, how to validate depends on the data keys
        val_results = validate(model, val_loader, device=config.device, epoch=epoch)
        # TODO: calculate several metrics: accuracy, precision, recall, f1-score and add them to result
        if 'tag_gts' in val_results and 'tag_preds' in val_results:
            tag_accuracy = [accuracy_score(val_results['tag_gts'], val_results['tag_preds'])]
            tag_precision, tag_recall, tag_f1, _ = precision_recall_fscore_support(val_results['tag_gts'], val_results['tag_preds'], average='weighted')
            logging.info(f"Epoch {epoch}, TAG Accuracy: {tag_accuracy}, precision: {tag_precision}, recall: {tag_recall}, f1: {tag_f1}")
        if 'cls_gts' in val_results and 'cls_preds' in val_results:
            cls_accuracy = accuracy_score(val_results['cls_gts'], val_results['cls_preds'])
            cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(val_results['cls_gts'], val_results['cls_preds'], average='weighted')
            logging.info(f"Epoch {epoch}, CLS Accuracy: {cls_accuracy}, precision: {cls_precision}, recall: {cls_recall}, f1: {cls_f1}")
        plot_val_results(val_results, epoch, config.plot_path, data_config.tags_map, data_config.cls_map)

        # save every
        if epoch % config.save_every == 0:
            ensure_dir_exists(config.save_path)
            torch.save(model.state_dict(), config.save_path)
