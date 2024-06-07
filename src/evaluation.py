"""
模型评估
"""
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ner_dataset import NERDataset

from model import BERT_CRF
from tqdm import tqdm

import torch


if __name__ == '__main__':
    model_path = "../models/model_epoch_2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT_CRF('bert-base-chinese', num_labels=9, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="./bert-base-chinese")  # load the pretrained model
    label_map = {'I_T': 8, 'I_PER': 7, 'B_LOC': 2, 'B_PER': 0, 'B_T': 3, 'B_ORG': 6, 'I_LOC': 4, 'O': 5, 'I_ORG': 1}

    # Initialize dataset and dataloader
    val_dataset = NERDataset('./data/dev.txt', './data/dev_TAG.txt', tokenizer, label_map=label_map, max_len=512)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=18)

    # Initialize lists to collect predictions and true labels
    all_preds = []
    all_labels = []

    # Iterate through the dataloader
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch["input_ids"], batch["attention_mask"])

            for pred, label, mask in zip(predictions, batch["labels"], batch["attention_mask"]):
                valid_labels = label[mask == 1].cpu().numpy()
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
    plt.savefig('confusion_matrix.png')
    plt.show()

