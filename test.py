"""
test.py
读取测试文件，生成预测结果
"""
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from model import BERT_CRF

CACHE_DIR = 'bert-base-chinese'
file_path = "data/dataset1/test.txt"
out_path = "../data/test_TAG.txt"
model_path = "../models/model_epoch_5.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_CRF(CACHE_DIR, num_labels=9, pretrained=False, cache_dir=CACHE_DIR).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
tokenizer = BertTokenizer.from_pretrained(CACHE_DIR, cache_dir=CACHE_DIR)  # load the pretrained model
label_map = {'I_T': 8, 'I_PER': 7, 'B_LOC': 2, 'B_PER': 0, 'B_T': 3, 'B_ORG': 6, 'I_LOC': 4, 'O': 5, 'I_ORG': 1}
idx2label = {idx: label for label, idx in label_map.items()}


max_len = 512


def inference(model, tokenizer, text):
    tokens = [tokenizer.tokenize(char)[0] for char in text]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)[0]

    tags = [idx2label[idx] for idx in outputs[1:-1]]
    return tags


with open(file_path, 'r', encoding='utf-8') as f:
    total_length = sum(1 for line in f)

with open(file_path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as f_out:
    for file_line in tqdm(f, total=total_length):
        file_line: str = file_line.strip()

        file_parts = file_line.split(' ')
        line_text = "".join(file_parts)

        if len(line_text) > max_len - 2:
            all_tags = []
            for sub_line in line_text.split('。'):
                assert len(sub_line) < 512 - 2
                tags = inference(model, tokenizer, sub_line)
                all_tags.extend(tags)
                all_tags.append('O')
            all_tags.pop(-1)
            line_out = " ".join(all_tags) + '\n'
        else:
            all_tags = inference(model, tokenizer, line_text)
            line_out = " ".join(all_tags) + '\n'

        assert len(file_parts) == len(all_tags)
        f_out.write(line_out)

