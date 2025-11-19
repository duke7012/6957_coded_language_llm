import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class ScoreObj:
    def __init__(self, score):
        self.score = score


def moverscore_embedding_similarity(ref_emb, pred_emb):
    cost_matrix = cdist(ref_emb, pred_emb, metric="cosine")
    n = len(ref_emb)
    m = len(pred_emb)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    transport_cost = (cost_matrix[row_ind, col_ind]).sum()
    return 1 - transport_cost / max(n, m)


def calc_moverscore(model, gold_lines, pred_lines):
    scores = []
    for g, p in zip(gold_lines, pred_lines):
        ref_emb = model.encode(g.split(), convert_to_numpy=True)
        pred_emb = model.encode(p.split(), convert_to_numpy=True)
        scores.append(ScoreObj(moverscore_embedding_similarity(ref_emb, pred_emb)))
    return [s.score for s in scores]


if __name__ == "__main__":
    gold_dir_str = "data/parsed/"
    pred_dir_str = "data/llama31_pred_partial_lora_dora/"
    output_dir = "data/mover_score_llama_loradora/"

    os.makedirs(output_dir, exist_ok=True)

    gold_files = sorted(os.listdir(gold_dir_str))
    pred_files = sorted(os.listdir(pred_dir_str))

    model = SentenceTransformer("all-mpnet-base-v2")

    for pred_file in pred_files:
        gold_file = pred_file

        with open(os.path.join(gold_dir_str, gold_file), "r", encoding="utf-8", errors="replace") as fh:
            gold_lines = [line.rstrip("\n") for line in fh]
            gold_lines = [re.sub(r"<pad>", "", line) for line in gold_lines]
            gold_lines = [re.sub(r"</s>", "", line) for line in gold_lines]
            gold_lines = [line.strip() for line in gold_lines]
            gold_lines = [line for line in gold_lines if line]

        with open(os.path.join(pred_dir_str, pred_file), "r", encoding="utf-8", errors="replace") as fh:
            pred_lines = [line.rstrip("\n") for line in fh]
            pred_lines = [line for line in pred_lines if line]

        if len(gold_lines) != len(pred_lines):
            raise ValueError(
                f"Line count mismatch for {pred_file}: "
                f"gold={len(gold_lines)}, pred={len(pred_lines)}"
            )

        ms_values = calc_moverscore(model, gold_lines, pred_lines)
        base = os.path.splitext(os.path.basename(gold_file))[0]
        filename = os.path.join(output_dir, base + "_scores.txt")

        df = pd.DataFrame({"MoverScore": ms_values})
        with open(filename, "w", encoding="utf-8") as f_out:
            f_out.write(f"Mean MoverScore for {base}: {df['MoverScore'].mean():.4f}\n")
            f_out.write(f"Max MoverScore for {base}: {df['MoverScore'].max():.4f}\n")
            f_out.write(f"Min MoverScore for {base}: {df['MoverScore'].min():.4f}\n")

        print(f"MoverScore stats written for {base}")

