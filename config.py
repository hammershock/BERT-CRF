import json
import os
from typing import Optional

from prettytable import PrettyTable


class _BaseConfig:
    def __repr__(self):
        return json.dumps({k: v for k, v in self.__dict__.items() if not isinstance(v, DataConfig)})

    def dump_to_json(self, json_path="./config.json"):
        if os.path.exists(json_path):  # avoid overwrite
            raise FileExistsError
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Configuration dumped to {json_path}")

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TrainerConfig(_BaseConfig):
    def __init__(self, *, bert_model_path: str, pretrained_model=None, num_hidden_layers=12,
                 num_epochs, batch_size, lr, lr_crf, device, num_workers=12,
                 save_path: Optional = None, plot_path: Optional = None, save_every=1, log_path: Optional = None):
        # model hyperparameters
        self.bert_model_path = bert_model_path  # bert-base-chinese model path
        self.pretrained_model = pretrained_model  # load model from checkpoint
        self.num_hidden_layers = num_hidden_layers

        # training hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_crf = lr_crf
        self.device = device  # device type
        self.num_workers = num_workers

        # save & plot & logging settings (Optional)
        self.save_path = save_path
        self.plot_path = plot_path
        self.save_every = save_every
        self.log_path = log_path

    def print_config(self):
        """use pretty to print the config out, is that cool?"""
        table = PrettyTable()
        table.field_names = ["Parameter", "Value"]
        for key, value in self.__dict__.items():
            table.add_row([key, value])
        print(table)


class DataConfig(_BaseConfig):
    def __init__(self, corpus_file, tags_file, cls_file):
        self.corpus_file = corpus_file
        self.tags_file = tags_file
        self.cls_file = cls_file


class DatasetConfig(_BaseConfig):
    def __init__(self, dataset_dir, max_seq_len, overlap, tags_map, special_tag, cls_map, data):
        self.dataset_dir = dataset_dir
        self.max_seq_len = max_seq_len
        self.overlap = overlap
        self.tags_map = tags_map
        self.special_tag = special_tag
        self.cls_map = cls_map
        self.data = data
        self.train_data = DataConfig(**data["train"])
        self.dev_data = DataConfig(**data["dev"])
        self.test_file = data["test"]


if __name__ == "__main__":
    data_config = DatasetConfig.from_json_file("data/product_comments/data.json")
    print(data_config)
    train_config = TrainerConfig.from_json_file("./data/product_comments/train_config.json")
