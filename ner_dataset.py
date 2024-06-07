import torch
from torch.utils.data import Dataset

from utils import check_file_pair


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
