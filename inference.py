import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datasets import NERDataset
from model import BERT_CRF
from tqdm import tqdm

import torch


if __name__ == '__main__':
    model_path = "./model_epoch_2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT_CRF('bert-base-chinese', num_labels=9, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./bert-base-chinese")  # load the pretrained model
    label_map = {'I_T': 0, 'I_PER': 1, 'B_LOC': 2, 'B_PER': 3, 'B_T': 4, 'B_ORG': 5, 'I_LOC': 6, 'O': 7, 'I_ORG': 8}
    # idx2label = {idx: label for label, idx in label_map.items()}
    # tokens = tokenizer.tokenize("北京大学是一所著名的高等学府")
    # tokens = ['[CLS]'] + tokens + ['[SEP]']
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     outputs = model(input_ids, attention_mask=torch.ones(input_ids.shape, dtype=torch.long).to(device))
    # print(outputs)
    # print([idx2label[idx] for idx in outputs[0][1:-1]])

    # Initialize dataset and dataloader
    val_dataset = NERDataset('./data/dev.txt', './data/dev_TAG.txt', tokenizer, label_map=label_map, max_len=512)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=18)

    # Initialize lists to collect predictions and true labels
    all_preds = []
    all_labels = []

    # Iterate through the dataloader
    for batch in tqdm(val_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions = model(batch["input_ids"], batch["attention_mask"])

        for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
            valid_labels = label[mask == 1]
            valid_preds = pred if isinstance(model, BERT_CRF) else pred[mask == 1]

            all_labels.extend(valid_labels)
            all_preds.extend(valid_preds)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')

    # Plot the normalized confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(label_map.keys()))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.show()

