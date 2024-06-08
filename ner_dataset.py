from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from joblib import Memory
from tqdm import tqdm

from utils import check_file_pair

memory = Memory("./.cache", verbose=0)


def load_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f.readlines()]


class NERDataset(Dataset):
    def __init__(self, max_seq_len, file_path, label_path, tokenizer, label_map, special_label_id):
        self.max_len = max_seq_len

        check_file_pair(file_path, label_path)
        self.data = load_txt_file(file_path)
        self.tags = load_txt_file(label_path)

        self.tokenizer = tokenizer
        self.label_map = label_map
        self.special_label_id = special_label_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx]
        tags = self.tags[idx]

        tokens = []
        label_ids = []

        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([self.label_map[tag]] * len(word_tokens))

        tokens = tokens[:self.max_len-2]
        label_ids = label_ids[:self.max_len-2]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [self.special_label_id] + label_ids + [self.special_label_id]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        label_ids = label_ids + ([self.special_label_id] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


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
def make_ner_dataset(max_seq_len, file_path, label_path, tokenizer, label_map, special_label_id, overlap=128):
    """load corpus lines and tokenize them into tensors"""
    # max_seq_len, file_path, label_path, tokenizer, label_map, special_label_id
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(file_path, 'r', encoding='utf-8') as f1, open(label_path, 'r', encoding='utf-8') as f2:
        batch_input_ids, attention_mask, batch_label_ids = [], [], []
        for line1, line2 in tqdm(zip(f1, f2), "building dataset", total=total_lines):
            input_tokens = []
            label_ids = []
            for part1, part2 in zip(line1.strip().split(), line2.strip().split()):
                tokens = tokenizer.tokenize(part1)
                input_tokens.extend(tokens)
                label_ids.extend([label_map[part2]] * len(tokens))

            input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            label_ids = [special_label_id] + label_ids + [special_label_id]

            assert len(input_ids) == len(label_ids)
            b_input_ids, b_attention_mask = _create_batches(input_ids, max_seq_len=max_seq_len, overlap=overlap, pad_id=tokenizer.pad_token_id)
            b_label_ids, _ = _create_batches(label_ids, max_seq_len=max_seq_len, overlap=overlap, pad_id=tokenizer.pad_token_id)
            batch_input_ids.append(b_input_ids)
            attention_mask.append(b_attention_mask)
            batch_label_ids.append(b_label_ids)

        batch_label_ids = torch.from_numpy(np.concatenate(batch_label_ids, dtype=np.int64))
        batch_input_ids = torch.from_numpy(np.concatenate(batch_input_ids, dtype=np.int64))
        attention_mask = torch.from_numpy(np.concatenate(attention_mask, dtype=np.int64))
        return TensorDataset(batch_input_ids, attention_mask, batch_label_ids)
