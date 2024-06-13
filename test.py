"""
test.py
读取测试文件，生成预测结果
"""
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from config import DatasetConfig, TrainerConfig
from model import BERT_CRF
from ner_dataset import make_dataset


def read_lines(file_path: str) -> List[str]:
    """
    读取CSV文件中的文本内容并返回一个字符串列表
    """
    df = pd.read_csv(file_path)
    text_lines = df['text'].tolist()
    return text_lines


@torch.no_grad()
def inference(model, inputs: dict, device) -> Tuple[np.ndarray, np.ndarray]:
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tag_ids, cls_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    cls = torch.argmax(cls_logits, dim=-1)
    classes = cls.cpu().numpy()  # 【batch, num_cls】

    return tag_ids, classes


def test(filepath, data_config_path, train_config_path, max_seq_len=128, overlap=0):
    """
    序列标注+文本分类模型测试过程
    """
    df = pd.read_csv("./data/product_comments/test_public.csv")
    text_lines = df['text'].tolist()
    config = TrainerConfig.from_json_file(train_config_path)
    data_config = DatasetConfig.from_json_file(data_config_path)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
    corpus_lines = read_lines(filepath)

    dataset = make_dataset(corpus_lines, tokenizer, max_seq_len, overlap)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = BERT_CRF(config.bert_model_path,
                     num_labels=len(data_config.tags_map) if data_config.tags_map else 1,
                     num_classes=len(data_config.cls_map) if data_config.cls_map else 1,
                     ).to(config.device)
    model.load_state_dict(torch.load(config.pretrained_model, map_location=config.device))
    model.eval()

    tag_idx2tag = {v: k for k, v in data_config.tags_map.items()}
    class_idx2class = {v: k for k, v in data_config.cls_map.items()}

    results = defaultdict(lambda: defaultdict(list))
    for batch in tqdm(dataloader, "testing lines"):
        tag_ids, classes = inference(model, batch, config.device)

        # tag_ids: np.ndarray [batch, seq_len]
        # classes: np.ndarray [batch]
        id_groups = batch["id_groups"].cpu().numpy()
        # id_groups: np.ndarray [batch]
        # 例如: [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4]
        # 原始的一个长句子可能被切分为不同片段，这在id_groups可以体现出来

        for i, group_id in enumerate(id_groups):
            tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
            tokens = [token for token in tokens if token != "[PAD]"]
            # print(group_id)
            # print("".join(tokens))
            # print(text_lines[group_id])
            line = list(text_lines[group_id])
            # if len(line) != len(tokens) - 2:
            #     for char, token in zip(line, tokens[1:-1]):
            #         print(char, token)
            #     print("-" * 10)
            results[group_id]['tags'].append(tag_ids[i])
            results[group_id]['classes'].append(classes[i])
            # print(type(tokens))
            # print(type(tag_ids[i]))
            # print(len(tokens), len(tag_ids[i]))
            assert len(tokens) == len(tag_ids[i])

    final_tags = {}
    final_classes = {}

    for group_id, data in results.items():
        # Flatten and remove overlap
        flat_tags = [tag_idx2tag[tag] for seq in data['tags'] for tag in seq[overlap:]]
        final_tags[group_id] = flat_tags[1:-1]

        # Determine final class by majority vote
        flat_classes = [class_idx2class[cls] for cls in data['classes']]
        final_class = max(set(flat_classes), key=flat_classes.count)
        final_classes[group_id] = final_class

    return final_tags, final_classes


def save_results_to_csv(final_tags: Dict[int, List[str]], final_classes: Dict[int, str], output_file: str):
    """
    将测试结果保存到CSV文件中
    """
    data = []
    for id in final_tags.keys():
        bio_anno = ' '.join(final_tags[id])
        cls = final_classes[id]
        data.append([id, bio_anno, cls])

    df = pd.DataFrame(data, columns=['id', 'BIO_anno', 'class'])
    df.to_csv(output_file, index=False)


def check_alignment(input_file: str, output_file: str) -> bool:
    """
    检查输入文件和输出文件是否满足每一行字字对应
    """
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    for idx in range(len(input_df)):
        input_text = input_df.iloc[idx]['text'].strip()
        output_tags = output_df.iloc[idx]['BIO_anno'].split()

        if len(input_text) != len(output_tags):
            print(f"Mismatch found in line {idx}:")
            print(f"Input text ({len(input_text)} chars): {input_text}")
            print(f"Output tags ({len(output_tags)} tags): {' '.join(output_tags)}")
            for char, tag in zip(input_text, output_tags):
                print(f"{char}: {tag}")
            # return False

    print("All lines are properly aligned.")
    return True


if __name__ == '__main__':
    final_tags, final_classes = test("data/product_comments/test_public.csv", "./data/product_comments/data.json",
         "./data/product_comments/train_config.json", max_seq_len=128)

    # final_tags: Dict[id, List[tags]]
    # final_classes: Dict[id, cls]

    save_results_to_csv(final_tags, final_classes, "./output.csv")
    check_alignment("data/product_comments/test_public.csv", "./output.csv")
