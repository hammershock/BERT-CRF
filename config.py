import json
import os

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
    def __init__(self, *, bert_model_path, device,
                 num_epochs: int, max_seq_len: int, overlap: int, batch_size: int, lr: float, lr_crf: float,
                 num_hidden_layers: int, save_path: str, plot_path: str, save_every: int, log_path: str,
                 num_workers: int, special_token_type: str, pretrained_model):
        self.bert_model_path = bert_model_path
        self.device = device
        self.special_token_type = special_token_type
        self.num_epochs = num_epochs
        self.max_seq_len = max_seq_len
        self.overlap = overlap
        self.batch_size = batch_size
        self.lr = lr
        self.lr_crf = lr_crf
        self.num_hidden_layers = num_hidden_layers
        self.save_path = save_path
        self.plot_path = plot_path
        self.save_every = save_every
        self.log_path = log_path
        self.num_workers = num_workers
        self.pretrained_model = pretrained_model

    def print_config(self):
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
    def __init__(self, dataset_dir, tags_map, special_tag, cls_map, data):
        self.dataset_dir = dataset_dir
        self.tags_map = tags_map
        self.special_tag = special_tag
        self.cls_map = cls_map
        self.data = data
        self.train_data = DataConfig(**data["train"])
        self.dev_data = DataConfig(**data["dev"])


if __name__ == "__main__":
    data_config = DatasetConfig.from_json_file("data/dataset1/data.json")
    print(data_config)
