# BERT-CRF

**序列标注**和**序列分类**训练以及评估流程。

![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-red)
![Transformers](https://img.shields.io/badge/Transformers-4.12.3-green)
![pytorch-crf](https://img.shields.io/badge/pytorch--crf-0.7.2-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.21.2-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.3-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blueviolet)
![joblib](https://img.shields.io/badge/joblib-1.1.0-ff69b4)


## 序列标注任务

![NER](https://github.com/hammershock/BERT-CRF/assets/109429530/c71475cd-0d10-41c2-8515-f33cffee609d)

序列标注任务可以认为是token级别的分类任务，需要将句子中的不同token分为不同类别。不同于普通的分类任务，句子中的序列元素之间并不具有独立性，因此不能通过传统的分类器独立地处理每个token并预测类别。我们采用条件随机场(CRF)处理序列标签之间的天然的相关关系，达到更加准确的序列标注。

### 序列标注任务举例：

- **命名实体识别(NER)**

  命名实体识别（NER，Named Entity Recognition）是自然语言处理中的一项基础任务，旨在从文本中识别出具有特定意义的实体，如人名、地名、组织名等。这里提供我在NLP课程中编写的命名实体识别的[实验报告](./document.pdf)。

- **句子分割(Text Split)**

  将句子分为若干个部分，比如分词任务，每个部分可以有一个标签。

### 句子分类任务举例:
- **句子情感分类**
  将句子按照情感分为两个类或多个类别

## 使用方法

### 1. 安装依赖库：

```bash
pip install torch transformers pytorch-crf numpy matplotlib scikit-learn prettytable joblib tqdm
```

### 2. 下载预训练模型和分词器：

从huggingface下载`bert-base-chinese`[预训练模型](https://huggingface.co/google-bert/bert-base-chinese)以及分词器。

### 3. 准备训练数据：

标签和文本均采用空格隔开，一一对应，且分开保存。每个空格隔开的部分不强制要求是一个token。

示例数据: 解压缩`data.rar`到项目目录。

### 4. 配置训练设置：

示例：
```json
{
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
  "plot_path": "./plots",
  "device": "cuda",
  "num_workers": 14,
  "pretrained_model": null
}
```

### 5. 配置数据集配置：

这个模型既可以用作序列标注模型，也可以用于句子分类，因为还有一个分类头。
你可以只用它序列标注或句子分类，也可以同时用作两个用途。其余的标签文件设置为空即可。

示例：
```json
{
  "dataset_dir": "./data/dataset1",
  "tags_map": {"I_T": 8, "I_PER": 7, "B_LOC": 2, "B_PER": 0, "B_T": 3, "B_ORG": 6, "I_LOC": 4, "O": 5, "I_ORG": 1},
  "special_tag": "O",
  "cls_map": null,  // 序列分类的标签映射表，同tags_map
  "data": {
    "train": {
      "corpus_file": "train.txt",
      "tags_file": "train_TAG.txt",
      "cls_file": null  // 设置为空，相当于只训练序列标注
    },
    "dev": {
      "corpus_file": "dev.txt",
      "tags_file": "dev_TAG.txt",
      "cls_file": null
    }
  }
}
```

### 6. 开始训练：

```bash
python train.py --train_config path/to/your/train/config --data_config path/to/your/data/config
```

### 7. 预测：

训练完成后，可以使用训练好的模型进行预测。请参考`inference.py`脚本。
