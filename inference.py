import torch
from transformers import BertTokenizer

from model import BERT_CRF
from config import TrainerConfig, DatasetConfig
from utils import load_json_file


@torch.no_grad()
def inference(text: str, config_path: str, data_config_path: str, bert_model_path: str, model_path: str):
    config = TrainerConfig.from_json_file(config_path)
    data_config = DatasetConfig.from_json_file(data_config_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    model = BERT_CRF(config.bert_model_path,
                     num_labels=len(data_config.tags_map) if data_config.tags_map else 1,
                     num_classes=len(data_config.cls_map) if data_config.cls_map else 1,
                     ).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    label_map_inv = {v: k for k, v in data_config.tags_map.items()}

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    output, cls = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    output = output[0][1:-1]
    tags = [label_map_inv[idx] for idx in output]
    cls_logits = cls[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]

    results = [(token, tag) for token, tag in zip(tokens, tags)]
    for token, tag in zip(tokens, tags):
        print(f"{token}\t{tag}")
    return results, cls_logits


if __name__ == '__main__':
    input_string = ("1月9日,记者在泽州县周村镇周村采访时,发现一通落款怪异的石碑。后查证得知,"
                    "这通刻满日寇士兵名字的石碑,是当年侵华日寇为纪念所谓的战死军人所立的“忠魂”碑,"
                    "其对侵略进行赤裸裸美化的行径,成为日本侵略中国的又一罪证。")

    config_path = "data/dataset1/train_config.json"
    data_config_path = "./data/dataset1/data.json"
    bert_model_path = './bert-base-chinese'
    model_path = "./models/model_product_comments.pth"

    inference(input_string, config_path, data_config_path, bert_model_path, model_path)
