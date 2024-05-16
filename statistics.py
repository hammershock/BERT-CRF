import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_data(file_path, title=""):
    assert file_path.endswith('.txt')
    counts = defaultdict(int)

    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split()
            for item in line:
                counts[item] += 1

    print(counts)  # Optional: for debugging purposes
    total = sum(counts.values())
    print(counts['O'] / total)

    # Visualizing the frequency histogram
    labels, values = zip(*counts.items())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Tags')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig(f'./figs/{title.replace(" ", "_")}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    visualize_data('./data/train_TAG.txt', "Frequency of Train Tags")
    visualize_data('./data/dev_TAG.txt', "Frequency of Dev Tags")

