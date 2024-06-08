import torch
from transformers import BertTokenizer

from model import BERT_CRF
from config import TrainerConfig
from utils import load_json_file


if __name__ == '__main__':
    config = TrainerConfig.from_json_file("./config/train_config.json")
    label_map = load_json_file("./config/label_vocab.json")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BERT_CRF("./bert-base-chinese", len(label_map)).to(config.device)
    model.load_state_dict(torch.load("./models/my_model.pth", map_location=config.device))
    model.eval()
    label_map_inv = {v: k for k, v in label_map.items()}
    input_string = "1月9日,记者在泽州县周村镇周村采访时,发现一通落款怪异的石碑。后查证得知,这通刻满日寇士兵名字的石碑,是当年侵华日寇为纪念所谓的战死军人所立的“忠魂”碑,其对侵略进行赤裸裸美化的行径,成为日本侵略中国的又一罪证。"

    with torch.no_grad():
        inputs = tokenizer(input_string, return_tensors="pt", padding=True)
        inputs = {k: v.to(config.device) for k, v in inputs.items()}
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0][1:-1]
        tags = [label_map_inv[idx] for idx in output]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]

    for token, tag in zip(tokens, tags):
        print(f"{token}\t{tag}")
