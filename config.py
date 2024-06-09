import json
import os

from prettytable import PrettyTable


class _BaseConfig:
    def __repr__(self):
        return json.dumps(self.__dict__)

    def dump_to_json(self, json_path="./config.json"):
        if os.path.exists(json_path):  # avoid overwrite
            raise FileExistsError
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Configuration dumped to {json_path}")

    @staticmethod
    def from_json_file(json_file) -> 'TrainerConfig':
        with open(json_file, 'r') as f:
            data = json.load(f)
        return TrainerConfig(**data)


class TrainerConfig(_BaseConfig):
    def __init__(self, *, train_path, train_label_path, val_path, val_label_path, bert_model_path, device,
                 num_epochs: int, max_seq_len: int, overlap: int, batch_size: int, lr: float, lr_crf: float,
                 num_hidden_layers: int, save_path: str, save_every: int, log_path: str, num_workers: int,
                 special_token_type: str):
        self.train_path = train_path
        self.train_label_path = train_label_path
        self.val_path = val_path
        self.val_label_path = val_label_path
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
        self.save_every = save_every
        self.log_path = log_path
        self.num_workers = num_workers

    def print_config(self):
        table = PrettyTable()
        table.field_names = ["Parameter", "Value"]

        for key, value in self.__dict__.items():
            table.add_row([key, value])

        print(table)
