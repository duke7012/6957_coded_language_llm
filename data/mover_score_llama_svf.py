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
    """
    Compute Moverscore between two embedding matrices using cosine distances.
    Uses Earth Mover Distance (Wasserstein-1).
    """

    cost_matrix = cdist(ref_emb, pred_emb, metric="cosine")

    n = len(ref_emb)
    m = len(pred_emb)
    weights_ref = np.ones(n) / n
    weights_pred = np.ones(m) / m

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    transport_cost = (cost_matrix[row_ind, col_ind]).sum()
    movescore = 1 - transport_cost / max(n, m)

    return movescore


def calc_moverscore(model, g_lines, p_lines):
    scores = []

    for g, p in zip(g_lines, p_lines):
        ref_emb = model.encode(g.split(), convert_to_numpy=True)
        pred_emb = model.encode(p.split(), convert_to_numpy=True)

        ms = moverscore_embedding_similarity(ref_emb, pred_emb)
        scores.append(ScoreObj(ms))

    float_scores = [s.score for s in scores]
    return scores, float_scores


if __name__ == "__main__":
    gold_dir_str = "data/parsed/"
    pred_dir_str = "data/llama31_pred_partial_svf/"

    gold_dir = sorted(os.listdir(gold_dir_str))
    pred_dir = sorted(os.listdir(pred_dir_str))

    output_dir = "data/mover_score_llama_svf/"
    os.makedirs(output_dir, exist_ok=True)

    model = SentenceTransformer("all-mpnet-base-v2")

    for pred_f in pred_dir:
        gold_f = pred_f

        with open(gold_dir_str + gold_f, "r", encoding="utf-8", errors="replace") as file:
            gold_lines = file.readlines()
            gold_lines = [line.rstrip("\n") for line in gold_lines]
            gold_lines = [re.sub(r"<pad>", "", line) for line in gold_lines]
            gold_lines = [re.sub(r"</s>", "", line) for line in gold_lines]
            gold_lines = [line.strip() for line in gold_lines]
            gold_lines = [string for string in gold_lines if string]

        with open(pred_dir_str + pred_f, "r", encoding="utf-8", errors="replace") as file:
            pred_lines = file.readlines()
            pred_lines = [line.rstrip("\n") for line in pred_lines]
            pred_lines = [string for string in pred_lines if string]

        if len(gold_lines) != len(pred_lines):
            raise ValueError(
                f"Line count mismatch for {pred_f}: gold={len(gold_lines)}, pred={len(pred_lines)}"
            )

        _, ms_vals = calc_moverscore(model, gold_lines, pred_lines)

        orig_filename = os.path.splitext(os.path.basename(gold_f))[0]
        filename = os.path.join(output_dir, orig_filename + "_scores.txt")

        score_df = pd.DataFrame()
        score_df["MoverScore"] = ms_vals

        score_txt = []
        score_txt.append(f"Mean MoverScore for {orig_filename}: {score_df['MoverScore'].mean():.4f}")
        score_txt.append(f"Max MoverScore for {orig_filename}: {score_df['MoverScore'].max():.4f}")
        score_txt.append(f"Min MoverScore for {orig_filename}: {score_df['MoverScore'].min():.4f}")

        with open(filename, "w", encoding="utf-8") as txt_file:
            for line in score_txt:
                txt_file.write(line + "\n")

        print(f"MoverScore stats written for {orig_filename}")
