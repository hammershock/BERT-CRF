import json
import os

from torch.utils.data import Dataset


def ensure_dir_exists(filepath) -> None:
    filedir = filepath if os.path.isdir(filepath) else os.path.dirname(filepath)
    os.makedirs(filedir, exist_ok=True)


def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def check_file_pair(source_path, target_path):
    """检查样本和标签是否字字对应"""
    with open(source_path, "r") as f1, open(target_path, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()
            parts1 = line1.split(" ")
            parts2 = line2.split(" ")

            if len(parts1) != len(parts2):
                raise Exception(f"sequence length does not match: {len(parts1)} != {len(parts2)}")
    print("Congratulations! Check Passed!")


class DictTensorDataset(Dataset):
    def __init__(self, **tensors):
        self.tensors = tensors
        assert all(tensors[key].size(0) == tensors[list(tensors.keys())[0]].size(0) for key in tensors), \
            "The first dimension of tensors must equal"

    def __len__(self):
        return self.tensors[list(self.tensors.keys())[0]].size(0)

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}
