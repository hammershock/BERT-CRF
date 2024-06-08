import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config import TrainerConfig
from model import BERT_CRF
from utils import load_json_file
from ner_dataset import make_ner_dataset
from train import validate, tqdm_iteration, ensure_dir_exists


def plot_confusion_matrix(all_labels, all_preds, plot_path):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')

    # Plot the normalized confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(label_map.keys()))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    ensure_dir_exists(plot_path)
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    config = TrainerConfig.from_json_file("./config/train_config.json")
    config.print_config()
    plot_path = "./plots/confusion_matrix.png"

    label_map = load_json_file("./config/label_vocab.json")
    model = BERT_CRF("./bert-base-chinese", num_labels=len(label_map), pretrained=0).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

    # Initialize dataset and dataloader
    val_dataset = make_ner_dataset(config.max_seq_len, config.val_path, config.val_label_path, tokenizer, label_map=label_map, special_label_id=label_map["O"])
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    result = tqdm_iteration(f"Validating", model, val_dataloader, None, config.device, validate)

    plot_confusion_matrix(result["all_labels"], result["all_predictions"], plot_path)
