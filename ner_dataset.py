import os.path
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import torch
from joblib import Memory
from tqdm import tqdm

from config import DataConfig, _DatasetConfig
from utils import DictTensorDataset

memory = Memory("./.cache", verbose=0)


def load_txt_file(filepath) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


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


def make_dataset(*, tokens: List[List[str]],
                 tags: Optional[List[List[int]]] = None,
                 classes: Optional[List[int]] = None, special_tag_id: Optional[int] = None,
                 tokenizer,
                 max_seq_len: int, overlap: int,
                 ) -> DictTensorDataset:
    # ================== check inputs ==================
    # assert tags is not None or classes is not None
    if tags is not None:
        assert len(tags) == len(tokens), f"{len(tags)} != {len(tokens)}"
        assert special_tag_id is not None
        for tokens_, tags_ in zip(tokens, tags):
            assert len(tokens_) == len(tags_), f"{len(tokens_)} != {len(tags_)}"
    else:
        tags = [None] * len(tokens)

    if classes is not None:
        assert len(classes) == len(tokens), f"{len(classes)} != {len(tokens)}"
    else:
        classes = [None] * len(tokens)
    # ==================================================

    data_out = defaultdict(list)

    for idx, (token_lst, tag_lst, cls) in tqdm(enumerate(zip(tokens, tags, classes)), total=len(tokens)):
        if tag_lst is None:
            tag_lst = [None] * len(token_lst)

        # tokenize each part of the line
        tokens = [("[CLS]", special_tag_id)]
        for token, tag in zip(token_lst, tag_lst):
            tokens += [(token, tag) for token in tokenizer.tokenize(token)]
        tokens += [("[SEP]", special_tag_id)]

        tokens, tag_ids = zip(*tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # break the sequence into equal parts, and add padding
        batched_input_ids, batched_attn_mask = _create_batches(input_ids, max_seq_len, overlap, tokenizer.pad_token_id)
        data_out["input_ids"].extend(batched_input_ids)
        data_out["attention_mask"].extend(batched_attn_mask)
        data_out["id_groups"].extend([idx] * len(batched_input_ids))

        if tag_ids[1] is not None:
            batched_tag_ids, _ = _create_batches(tag_ids, max_seq_len, overlap, tokenizer.pad_token_id)
            data_out["tag_ids"].extend(batched_tag_ids)

        if cls is not None:
            data_out["cls_ids"].extend([cls] * len(batched_input_ids))

    kwargs = {k: torch.from_numpy(np.array(v, dtype=np.int64)) for k, v in data_out.items()}

    return DictTensorDataset(**kwargs)


@memory.cache
def make_dataset_from_config(data_files: _DatasetConfig, config: DataConfig, tokenizer):
    """
    make tensor datasets from data config, with line cut and auto padding
    the out keys depends on the config
    """
    corpus_lines = load_txt_file(os.path.join(config.dataset_dir, data_files.corpus_file))
    tags_lines = load_txt_file(os.path.join(config.dataset_dir, data_files.tags_file)) if data_files.tags_file else None
    cls_lines = load_txt_file(os.path.join(config.dataset_dir, data_files.cls_file)) if data_files.cls_file else None

    return make_dataset(tokens=[line.strip().split(config.tag_sep) for line in corpus_lines],
                        tags=[[config.tags_map[tag] for tag in line.strip().split(config.tag_sep)] for line in tags_lines] if tags_lines else None,
                        classes=[int(line.strip()) for line in cls_lines] if cls_lines else None,
                        special_tag_id=config.tags_map.get(config.special_tag, None),
                        tokenizer=tokenizer,
                        max_seq_len=config.max_seq_len,
                        overlap=config.overlap,
                        )
