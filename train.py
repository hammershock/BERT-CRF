# -*- coding: utf-8 -*-
# train.py
"""
基于BERT+CRF的命名实体识别(NER)任务
———————— A simple Practice
@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn

"""
import argparse
import time
from collections import defaultdict
from typing import Dict, Any

import torch
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as multi_scores
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config import TrainerConfig, DataConfig
from fgm_attack import FGM
from model import BertCRF
from ner_dataset import make_dataset_from_config
from utils import ensure_dir_exists, with_tqdm, log_yield_results


@with_tqdm("Training")
@log_yield_results
def train_epoch(model, data_loader, optimizer, device, *, fgm=None, epoch=None, save_interval=None, save_path=None) -> \
        Dict[str, Any]:
    running_loss = 0.0
    model.train()
    t = time.time()  # last save time
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

        if save_interval is not None:
            t1 = time.time()
            if t1 - t > save_interval:
                torch.save(model.state_dict(), save_path)
                t = t1

        optimizer.step()
        running_loss += loss.item()
        yield {"running_loss": running_loss / (idx + 1)}


@with_tqdm("Validating")
@log_yield_results
@torch.no_grad()
def validate(model: BertCRF, data_loader, device, *, epoch=None) -> Dict[str, Any]:
    """
    validate model on dev set
    :param model:
    :param data_loader:
    :param device:
    :param epoch: current epoch progress
    :return:
    """
    model.eval()
    counts = defaultdict(list)
    ret = {}
    for idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model.forward(batch["input_ids"], batch["attention_mask"])

        if "tag_ids" in batch:  # 评估序列标注
            for label_pred, label, mask in zip(output.labels, batch["labels"], batch["attention_mask"]):
                valid_labels = label[mask == 1].detach().cpu().numpy()
                counts['tag_gts'].extend(valid_labels)
                counts['tag_preds'].extend(label_pred)
            ret["tag_accuracy"] = accuracy_score(counts["tag_gts"], counts["tag_preds"])
            ret["tag_precision"], ret["tag_recall"], ret["tag_f1"], _ = multi_scores(counts["tag_gts"],
                                                                                     counts["tag_preds"],
                                                                                     average='weighted')

        if "cls_ids" in batch:  # 评估分类任务
            cls_probs = output.cls_probs
            cls_preds = torch.argmax(cls_probs, dim=1).detach().cpu().numpy()
            cls_gts = batch["cls_ids"].detach().cpu().numpy()
            counts['cls_gts'].extend(cls_gts)
            counts['cls_preds'].extend(cls_preds)
            ret["cls_accuracy"] = accuracy_score(counts["cls_gts"], counts["cls_preds"])
            ret["cls_precision"], ret["cls_recall"], ret["cls_f1"], _ = multi_scores(counts["cls_gts"],
                                                                                     counts["cls_preds"],
                                                                                     average='weighted')

        yield ret


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Dataset configuration")
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--data_config', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # load configs
    train_config = TrainerConfig.from_yaml_file(args.train_config)
    print(train_config.table())

    # initialize logger
    ensure_dir_exists(train_config.log_path)
    logger.remove()
    logger.add(train_config.log_path, rotation="500 MB")
    logger.info(train_config)

    data_config = DataConfig.from_yaml_file(args.data_config)
    ensure_dir_exists(train_config.save_path)
    """
    Text labeling and Text classification, two tasks all in one model!
    You can train the model using either kind of data
    """
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(train_config.bert_model_path)

    # define model
    model = BertCRF(train_config.bert_model_path, num_labels=len(data_config.tags)).to(train_config.device)

    # load model from checkpoint
    if train_config.pretrained_model is not None:
        try:
            model.load_state_dict(torch.load(train_config.pretrained_model, map_location=train_config.device))
        except:
            pass

    # datasets
    train_set = make_dataset_from_config(data_config.train_data, data_config, tokenizer)
    val_set = make_dataset_from_config(data_config.dev_data, data_config, tokenizer)

    # prepare for training here
    # dataloaders, optimizer and logger
    train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True,
                              num_workers=train_config.num_workers)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size, shuffle=False,
                            num_workers=train_config.num_workers)

    optimizer = AdamW([
        {'params': list(model.bert.parameters()) + list(model.fc.parameters()), 'lr': train_config.lr},
        {'params': list(model.crf.parameters()), 'lr': train_config.lr_crf}
    ])

    fgm = FGM(model, epsilon=1.0) if train_config.use_fgm else None  # fgm attacker for embedding layers

    for epoch in range(train_config.num_epochs):
        # train the model with batched data, how to train depends on the data keys
        train_results = train_epoch(model, train_loader, optimizer, device=train_config.device, fgm=fgm, epoch=epoch,
                                    save_interval=train_config.save_interval, save_path=train_config.save_path)
        # model validation, how to validate depends on the data keys
        val_results = validate(model, val_loader, device=train_config.device, epoch=epoch)

        # save every
        if epoch % train_config.save_every == 0:
            torch.save(model.state_dict(), train_config.save_path)
