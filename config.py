"""
We use yaml format to store the configs, It is really nice.
"""
import json

import yaml


class _BaseConfig:
    def __repr__(self):
        return json.dumps({k: v for k, v in self.__dict__.items() if not isinstance(v, _DatasetConfig)})

    @classmethod
    def from_yaml_file(cls, yaml_file):
        assert yaml_file.endswith(('.yml', '.yaml')), 'Only .yml and .yaml files are supported'
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class DataConfig(_BaseConfig):
    def __init__(self, *, tags, special_tag, tag_sep=" ",  # data labels
                 dataset_dir, data,  # data paths
                 max_seq_len, overlap):  # pre-process
        self.tags = tags
        self.tags_map = {tag: idx for idx, tag in enumerate(tags)}
        self.special_tag = special_tag
        self.tag_sep = tag_sep

        # data paths
        self.dataset_dir = dataset_dir
        self.data = data
        self.train_data = _DatasetConfig(**data["train"])
        self.dev_data = _DatasetConfig(**data["dev"])

        self.max_seq_len = max_seq_len
        self.overlap = overlap


class _DatasetConfig(_BaseConfig):
    def __init__(self, corpus_file, tags_file=None, cls_file=None):
        self.corpus_file = corpus_file
        self.tags_file = tags_file
        self.cls_file = cls_file


class TrainerConfig(_BaseConfig):
    def __init__(self, *, bert_model_path, num_hidden_layers=12, device, num_workers,
                 num_epochs, batch_size, lr, lr_crf, use_fgm=False,
                 load_from_checkpoint_path=None, save_path=None, save_every=None, save_interval=None, log_path=None):
        self.bert_model_path = bert_model_path
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.num_workers = num_workers

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_crf = lr_crf
        self.use_fgm = use_fgm

        self.load_from_checkpoint_path = load_from_checkpoint_path
        self.save_path = save_path
        self.save_every = save_every
        self.save_interval = save_interval
        self.log_path = log_path

    def table(self):
        """use prettytable to print the config out, is that cool?"""
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = ["Parameter", "Value"]
        for key, value in self.__dict__.items():
            table.add_row([key, value])

        return table
