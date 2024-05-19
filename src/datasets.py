import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, data_file, tag_file, tokenizer, max_len=128, label_map=None):
        self.data = self.read_file(data_file)
        self.tags = self.read_file(tag_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = self.create_label_map() if label_map is None else label_map
        self.counts = self.calculate_label_frequencies()  # 统计标签频率

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().split() for line in f.readlines()]

    def create_label_map(self):
        labels = set(tag for tag_list in self.tags for tag in tag_list)
        return {label: i for i, label in enumerate(labels)}

    def calculate_label_frequencies(self):
        counts = {label: 0 for label in self.label_map.keys()}
        for tag_list in self.tags:
            for tag in tag_list:
                counts[tag] += 1
        return counts

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
        label_ids = [self.label_map['O']] + label_ids + [self.label_map['O']]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        label_ids = label_ids + ([self.label_map['O']] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def device_summary():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    memory_reserved = torch.cuda.memory_reserved()
    memory_allocated = torch.cuda.memory_allocated()

    memory_stats = torch.cuda.memory_stats()
    memory_free = memory_stats['reserved_bytes.all.freed'] + memory_stats['active_bytes.all.current']

    # 输出显存信息
    print(f'Total memory: {total_memory / 1024 / 1024:.3f} MB')
    print(f'Memory reserved: {memory_reserved / 1024 / 1024:.3f} MB')
    print(f'Memory allocated: {memory_allocated / 1024 / 1024:.3f} MB')
