import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from utils import ensure_dir_exists


def plot_auc_curve(true_labels, pred_probs, file_path, class_names):
    if not true_labels:
        return
    idx2name = {v: k for k, v in class_names.items()}
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_names)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i, color in zip(range(n_classes), ['blue', 'red', 'green', 'darkorange']):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {idx2name[i]} (area = {roc_auc[i]:0.2f})')

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(file_path)
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, plot_path, label_map):
    if not all_labels:
        return
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

