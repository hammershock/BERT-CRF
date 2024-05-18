# 基于Transformer的命名实体识别

三种模型结构：
1. 编码器BERT
2. 解码器GPT
3. 编码器+解码器T5

- 层数超参数自选，
- 随机初始化/预训练
- 有CRF，无CRF

提交：

- [x]: 
1. 文档（标签集，模型种类，层数，token数，初始字向量来源，向量维数，训练算法，学习率，batch大小，训练epoch数）
2. （损失曲线，准确率曲线，每轮epoch记录损失，
3. （重要）参考论文！大模型辅助程度！

模型结构（流程图）

train tags freq: {'O': 17182664, 'B_T': 180819, 'I_T': 494698, 'B_LOC': 206640, 'I_LOC': 326891, 'B_ORG': 15081, 'I_ORG': 33203, 'B_PER': 182664, 'I_PER': 352243}
dev tags freq: {'O': 2238962, 'B_LOC': 28221, 'I_LOC': 44605, 'B_T': 22789, 'I_T': 61373, 'B_PER': 24871, 'I_PER': 48577, 'B_ORG': 1851, 'I_ORG': 4030}

其中标签'O'占比超过90%，样本分布严重不均衡

训练设备：
单卡L20,显存48GB
batch size 420，10 epochs

Number of layers: 12
Vocabulary size: 21128
Embedding dimension: 768

Why CRF?
每个标签的出现并非独立

1. T5
2. BERT

附上test的预测结果文件：
test_TAG.txt

表1：标签含义

大模型使用：
User
将以下数据绘制成图表。我将提供给你两组数据，来源于相同模型的训练日志，用于命名实体识别，他们架构都是BERT+CRF，一个BERT加载bert-base-chinese预训练权重，一个随机初始化。分别训练10个epoch，记录了训练期间的Training Loss, Validating Loss,和验证集准确率。我希望你能给出分别给出loss和accuracy的两张图。
首先是第一组数据（预训练权重）：
。。。
然后是第二组数据(随机初始化)：
。。。
ChatGPT:
[Analysing]
chart