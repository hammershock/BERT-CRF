import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import json


def extract_data(filepath):
    filename = os.path.basename(filepath)
    basename = os.path.splitext(filename)[0]
    stats = defaultdict(int)
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            for item1, item2 in zip(line[:-1], line[1:]):
                stats[str((item1, item2))] += 1
    json.dump(stats, open(f'./figs/stats_{basename}.json', 'w'))
    print(stats)
    return stats


if __name__ == '__main__':
    extract_data('./data/train_TAG.txt')

