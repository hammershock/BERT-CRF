from transformers import BertTokenizer

from model import BERT_CRF

import torch


if __name__ == '__main__':
    model_path = "./model_epoch_2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT_CRF('bert-base-chinese', num_labels=9, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./bert-base-chinese")  # load the pretrained model
    label_map = {'I_T': 0, 'I_PER': 1, 'B_LOC': 2, 'B_PER': 3, 'B_T': 4, 'B_ORG': 5, 'I_LOC': 6, 'O': 7, 'I_ORG': 8}
    idx2label = {idx: label for label, idx in label_map.items()}
    tokens = tokenizer.tokenize("北京大学是一所著名的高等学府")
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=torch.ones(input_ids.shape, dtype=torch.long).to(device))
    print(outputs)
    print([idx2label[idx] for idx in outputs[0][1:-1]])
