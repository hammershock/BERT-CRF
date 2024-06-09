# BERT-CRF

本repo旨在提供一个使用BERT+CRF模型进行**序列标注**的通用模板，遵循简洁容易使用的原则。

![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-red)
![Transformers](https://img.shields.io/badge/Transformers-4.12.3-green)
![pytorch-crf](https://img.shields.io/badge/pytorch--crf-0.7.2-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.21.2-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.3-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blueviolet)
![prettytable](https://img.shields.io/badge/prettytable-2.4.0-lightgrey)
![joblib](https://img.shields.io/badge/joblib-1.1.0-ff69b4)
![tqdm](https://img.shields.io/badge/tqdm-4.62.3-9cf)


## 序列标注任务

序列标注任务可以认为是token级别的分类任务，需要将句子中的不同token分为不同类别。不同于普通的分类任务，句子中的序列元素之间并不具有独立性，因此不能通过传统的分类器独立地处理每个token并预测类别。我们采用条件随机场(CRF)处理序列标签之间的天然的相关关系，达到更加准确的序列标注。

### 序列标注任务举例：

- **命名实体识别(NER)**

  命名实体识别（NER，Named Entity Recognition）是自然语言处理中的一项基础任务，旨在从文本中识别出具有特定意义的实体，如人名、地名、组织名等。

- **句子分割(Text Split)**

  将句子分为若干个部分，比如分词任务，每个部分可以有一个标签。

## 使用方法

### 1. 安装依赖库：

```bash
pip install torch transformers pytorch-crf numpy matplotlib scikit-learn prettytable joblib tqdm
```

### 2. 下载预训练模型和分词器：

从huggingface下载`bert-base-chinese`预训练模型以及分词器。

### 3. 准备训练数据：

标签和文本均采用空格隔开，一一对应，且分开保存。每个空格隔开的部分不强制要求是一个token。

示例数据: 解压缩`data.rar`到项目目录。

### 4. 配置训练设置：

在`./config/train_config.json`文件中配置训练设置。

示例：
```json
{
  "train_path": "./data/train.txt",
  "train_label_path": "./data/train_TAG.txt",
  "val_path": "./data/dev.txt",
  "val_label_path": "./data/dev_TAG.txt",
  "bert_model_path": "./bert-base-chinese",
  "special_token_type": "O",
  "num_epochs": 50,
  "max_seq_len": 128,
  "overlap": 0,
  "batch_size": 280,
  "lr": 5e-5,
  "lr_crf": 5e-3,
  "num_hidden_layers": 12,
  "save_path": "./models/my_model.pth",
  "save_every": 1,
  "log_path": "./logs/train.log",
  "device": "cuda",
  "num_workers": 14
}
```

### 5. 配置标签词汇表：

在`./config/label_vocab.json`文件中配置标签词汇表。

示例：
```json
{"I_T": 8, "I_PER": 7, "B_LOC": 2, "B_PER": 0, "B_T": 3, "B_ORG": 6, "I_LOC": 4, "O": 5, "I_ORG": 1}
```

### 6. 开始训练：

```bash
python train.py
```

### 7. 预测和评估：

训练完成后，可以使用训练好的模型进行预测和评估。请参考`inference.py`和`evaluate.py`脚本。
