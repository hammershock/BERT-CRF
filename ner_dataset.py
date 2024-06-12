import os.path
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
from joblib import Memory

from utils import DictTensorDataset
from tqdm import tqdm

memory = Memory("./.cache", verbose=0)


def load_txt_file(filepath) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


def tokenize(text: str, tokenizer) -> List[int]:
    """Given a vocabulary, Tokenize a sentence using jieba and map the result into ids"""
    tokens = []
    for part in text.split():
        t = tokenizer.tokenize(part)
        assert len(t) == 1
        tokens.extend(t)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return input_ids


def decode(sequence: Sequence[int], vocab: Dict[str, int], unk_flag="[UNK]") -> str:
    """Given the vocabulary and a sequence of ids, decode it back to string"""
    idx2word = {v: k for k, v in vocab.items()}
    tokens = [idx2word.get(i, unk_flag) for i in sequence]
    return "".join(tokens)


def _create_batches(input_ids, max_seq_len: int, overlap: int, pad_id: int) -> Tuple[np.ndarray, ...]:
    """cut the line into pieces to create batches"""
    stride = max_seq_len - overlap
    num_batches = (len(input_ids) + stride - 1) // stride  # ceil(len(token_ids) / stride)
    batch_input_ids = np.full((num_batches, max_seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((num_batches, max_seq_len), dtype=np.int32)

    for i in range(num_batches):
        start_idx = i * stride
        end_idx = min(start_idx + max_seq_len, len(input_ids))
        batch_input_ids[i, :end_idx - start_idx] = input_ids[start_idx:end_idx]
        attention_mask[i, :end_idx - start_idx] = 1

    return batch_input_ids, attention_mask


@memory.cache
def make_ner_dataset(data_files, config, tokenizer):
    """
    make tensor datasets from data config, with line cut and auto padding
    the out keys depends on the config
    """
    data_in = {"documents": load_txt_file(os.path.join(config.dataset_dir, data_files.corpus_file))}
    nothings = [None] * len(data_in["documents"])
    # align the tags and cls labels to the corpus file
    data_in["sequences_labels"] = load_txt_file(os.path.join(config.dataset_dir, data_files.tags_file)) if data_files.tags_file else nothings
    data_in["sequences_cls"] = load_txt_file(os.path.join(config.dataset_dir, data_files.cls_file)) if data_files.cls_file else nothings

    data_out = defaultdict(list)
    zipped_gen = zip(data_in["documents"], data_in["sequences_labels"], data_in["sequences_cls"])
    for text, seq_labels, seq_cls in tqdm(zipped_gen, total=len(data_in["documents"])):
        parts = text.strip().split()
        labels = seq_labels.strip().split() if seq_labels is not None else [None] * len(parts)

        tokens = [("[CLS]", config.special_tag)]
        for part, label in zip(parts, labels):
            tokens += [(token, label) for token in tokenizer.tokenize(part)]
        tokens += [("[SEP]", config.special_tag)]

        tokens, label_ids = zip(*tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # cut the sequence into equal-length parts, and pad the remaining part
        b_input_ids, b_attention_mask = _create_batches(input_ids, max_seq_len=config.max_seq_len, overlap=config.overlap,
                                                        pad_id=tokenizer.pad_token_id)
        data_out["input_ids"].extend(b_input_ids)
        data_out["attention_mask"].extend(b_attention_mask)

        if seq_labels is not None:
            label_ids = [config.tags_map[label] for label in label_ids]

            b_label_ids, _ = _create_batches(label_ids, max_seq_len=config.max_seq_len, overlap=config.overlap,
                                             pad_id=tokenizer.pad_token_id)

            data_out["labels"].extend(b_label_ids)

        if seq_cls is not None:
            cls = seq_cls.strip()
            data_out["classes"].extend([config.cls_map[cls]] * len(b_input_ids))

    kwargs = {k: torch.from_numpy(np.array(v, dtype=np.int64)) for k, v in data_out.items()}

    return DictTensorDataset(**kwargs)

