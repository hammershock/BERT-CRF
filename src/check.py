"""
检查样本和标签是否字字对应
"""

source_path = "./data/test.txt"
target_path = "./2021213368.txt"

with open(source_path, "r") as f1, open(target_path, "r") as f2:
    for line1, line2 in zip(f1, f2):
        line1 = line1.strip()
        line2 = line2.strip()
        parts1 = line1.split(" ")
        parts2 = line2.split(" ")

        assert len(parts1) == len(parts2), (len(parts1), len(parts2))
