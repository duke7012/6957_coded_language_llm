import os
import sys
from collections import defaultdict

def parse_metric_line(line):
    """
    Extract the label and numeric value from a line like:
    'Mean Precision for D1_5_word_groups: 0.9374'
    Returns ('Mean Precision', 0.9374)
    """
    label_part, value_part = line.split(":")
    label = label_part.split(" for ")[0].strip()
    value = float(value_part.strip())
    return label, value

def aggregate_directory(dir_path):
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        full_path = os.path.join(dir_path, filename)

        if not os.path.isfile(full_path):
            continue  # skip subdirectories

        with open(full_path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                label, value = parse_metric_line(line)
                metric_sums[label] += value
                metric_counts[label] += 1

    # Compute averages
    averages = {label: metric_sums[label] / metric_counts[label]
                for label in metric_sums}

    return averages

def write_summary(dir_path, averages):
    output_path = os.path.join(dir_path, "overall_averages.txt")
    with open(output_path, "w") as out:
        for label in sorted(averages):
            out.write(f"{label}: {averages[label]:.6f}\n")
    print(f"Averages written to {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 aggregate_metrics.py <directory>")
        sys.exit(1)

    dir_path = sys.argv[1]

    if not os.path.isdir(dir_path):
        print(f"Error: '{dir_path}' is not a directory.")
        sys.exit(1)

    averages = aggregate_directory(dir_path)
    write_summary(dir_path, averages)

if __name__ == "__main__":
    main()


