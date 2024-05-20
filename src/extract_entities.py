import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

file_path = "../data/test.txt"
tag_path = "../data/test_TAG.txt"

# 定义只匹配中文字符的正则表达式
chinese_char_pattern = re.compile(r'[\u4e00-\u9fa5]')

def filter_chinese_chars(s):
    """只保留字符串中的中文字符"""
    return ''.join(ch for ch in s if chinese_char_pattern.match(ch))

# 读取文件并提取实体
with open(file_path, 'r') as f, open(tag_path, 'r') as g:
    entities = defaultdict(list)
    for file_line, label_line in zip(f, g):
        file_line = file_line.strip()
        label_line = label_line.strip()

        chars = file_line.split()
        tags = label_line.split()

        for char, tag in zip(chars, tags):
            if tag.startswith("B"):
                entity_type = tag.split("_")[-1]
                entities[entity_type].append([char])
            elif tag.startswith("I"):
                entity_type = tag.split("_")[-1]
                assert len(entities[entity_type])
                entities[entity_type][-1].append(char)

    entities = {k: ["".join(item) for item in v] for k, v in entities.items()}
    # 过滤非中文字符并统计词频
    word_counts = {k: Counter(filter_chinese_chars(string) for string in v if filter_chinese_chars(string)) for k, v in entities.items()}

# 绘制词云图
for entity_type, counter in word_counts.items():
    wordcloud = WordCloud(font_path='../data/fonts/AlimamaDaoLiTi.ttf', width=800, height=400).generate_from_frequencies(counter)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.tight_layout(pad=0)
    # plt.title(f'Word Cloud for {entity_type}')
    plt.axis("off")
    plt.savefig("../figs/" + entity_type + ".png")
    plt.show()
    plt.close()
