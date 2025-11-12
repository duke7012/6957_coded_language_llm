import os
import re

import pandas as pd
from bert_score import score
import torch


def calc_bert(g_lines, p_lines):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64 if device == "cuda" else 32

    P, R, F1 = score(
        g_lines,
        p_lines,
        lang="en",
        verbose=False,
        batch_size=batch_size,
        device=device,
    )
    return P.tolist(), R.tolist(), F1.tolist()


if __name__ == "__main__":
    gold_dir_str = "data/parsed/"
    pred_dir_str = "data/llama31_pred_partial_dora/"

    gold_dir = sorted(os.listdir(gold_dir_str))
    pred_dir = sorted(os.listdir(pred_dir_str))

    for pred_f in pred_dir:
        gold_f = pred_f

        with open(gold_dir_str + gold_f, "r") as file:
            gold_lines = file.readlines()
            gold_lines = [line.rstrip("\n") for line in gold_lines]
            gold_lines = [re.sub(r"<pad>", "", line) for line in gold_lines]
            gold_lines = [re.sub(r"</s>", "", line) for line in gold_lines]
            gold_lines = [line.strip() for line in gold_lines]
            gold_lines = [string for string in gold_lines if string]

        with open(pred_dir_str + pred_f, "r") as file:
            pred_lines = file.readlines()
            pred_lines = [line.rstrip("\n") for line in pred_lines]
            pred_lines = [string for string in pred_lines if string]

        p, r, f1 = calc_bert(gold_lines, pred_lines)

        orig_filename = os.path.basename(gold_f)
        orig_filename = os.path.splitext(orig_filename)[0]
        filename = "data/bert_score_llama_dora/" + orig_filename + "_scores.txt"

        score_df = pd.DataFrame()
        score_df["P"] = p
        score_df["R"] = r
        score_df["F1"] = f1

        score_txt = []
        score_txt.append("Mean Precision for {}: {:.4f}".format(orig_filename, score_df["P"].mean()))
        score_txt.append("Max Precision for {}: {:.4f}".format(orig_filename, score_df["P"].max()))
        score_txt.append("Min Precision for {}: {:.4f}".format(orig_filename, score_df["P"].min()))
        score_txt.append("Mean Recall for {}: {:.4f}".format(orig_filename, score_df["R"].mean()))
        score_txt.append("Max Recall for {}: {:.4f}".format(orig_filename, score_df["R"].max()))
        score_txt.append("Min Recall for {}: {:.4f}".format(orig_filename, score_df["R"].min()))
        score_txt.append("Mean F1 for {}: {:.4f}".format(orig_filename, score_df["F1"].mean()))
        score_txt.append("Max F1 for {}: {:.4f}".format(orig_filename, score_df["F1"].max()))
        score_txt.append("Min F1 for {}: {:.4f}".format(orig_filename, score_df["F1"].min()))

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as txt_file:
            for line in score_txt:
                txt_file.write(line + "\n")

