import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import json


def extract_data(filepath):
    filename = os.path.basename(filepath)
    basename = os.path.splitext(filename)[0]
    stats = defaultdict(int)
    max_len = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            if len(line) > max_len:
                max_len = len(line)
            for item1, item2 in zip(line[:-1], line[1:]):
                stats[str((item1, item2))] += 1
    json.dump(stats, open(f'./figs/stats_{basename}.json', 'w'))
    print(stats)
    print(max_len)
    return stats


def plot_stats(filepath):
    filename = os.path.basename(filepath)
    basename = os.path.splitext(filename)[0]

    sentence_lengths = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            length = len(line)
            sentence_lengths.append(length)

    # Calculate percentiles
    percentiles = [95, 99, 99.95]
    percentile_values = np.percentile(sentence_lengths, percentiles)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=range(1, max(sentence_lengths) + 2), edgecolor='black')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title(f'Sentence Length Distribution for {basename}')
    plt.grid(True)

    # Annotate percentiles
    for perc, value in zip(percentiles, percentile_values):
        plt.axvline(x=value, color='r', linestyle='--', label=f'{perc}th Percentile: {int(value)}')
        plt.text(value, plt.ylim()[1] * 0.9, f'{perc}%: {int(value)}', rotation=90, verticalalignment='center',
                 color='r')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_stats('./data/train_TAG.txt')

