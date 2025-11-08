import itertools
import os
import re

import pandas as pd
from bert_score import score


class ScoreObj:
    def __init__(self, P, R, F1):
        self.P = P
        self.R = R
        self.F1 = F1

def calc_bert(g_lines, p_lines):
    # Example texts
    reference = "This is a reference text example."
    candidate = "This is a candidate text example."
    scores = []
    # BERTScore calculation
    # scorer = BERTScorer(model_type='bert-base-uncased')
    for g, p in zip(g_lines, p_lines):
        P, R, F1 = score([g], [p], lang="en", verbose=True)
        scores.append(ScoreObj(P, R, F1))

    p = []
    r = []
    f1 = []
    for s in scores:
        p.append(s.P.mean())
        r.append(s.R.mean())
        f1.append(s.F1.mean())
        # print(f"BERTScore Precision: {s.P.mean():.4f}, Recall: {s.R.mean():.4f}, F1: {s.F1.mean():.4f}")
    
    return scores, p, r, f1


if __name__ == '__main__':
    
    gold_dir_str = "data/parsed/"
    pred_dir_str = "data/aya_pred_partial_dora/"

    gold_dir = sorted(os.listdir(gold_dir_str))
    pred_dir = sorted(os.listdir(pred_dir_str))

    # gold_file = "data/parsed/D1_5_word_groups.txt"
    # pred_file = "data/aya_pred/D1_5_word_groups.txt"
    
    for pred_f in pred_dir:
        gold_f = pred_f
        # if os.path.basename(gold_f) != os.path.basename(pred_f):
        # if not os.path.basename(pred_f).startswith(os.path.basename(pred_f)):
        #     print(f"{gold_f} != {pred_f}")
        #     break
        #
        # print(gold_f, pred_f)

        with open(gold_dir_str + gold_f, 'r') as file:
            gold_lines = file.readlines()
            # Remove trailing newline characters from each line
            gold_lines = [line.rstrip('\n') for line in gold_lines]
            gold_lines = [re.sub(r"<pad>", "", line) for line in gold_lines]
            gold_lines = [re.sub(r"</s>", "", line) for line in gold_lines]
            gold_lines = [line.strip() for line in gold_lines]
            gold_lines = [string for string in gold_lines if string]

        with open(pred_dir_str + pred_f, 'r') as file:
            pred_lines = file.readlines()
            # Remove trailing newline characters from each line
            pred_lines = [line.rstrip('\n') for line in pred_lines]
            pred_lines = [string for string in pred_lines if string]
        

        scores, p, r, f1 = calc_bert(gold_lines, pred_lines)
        
        orig_filename = os.path.basename(gold_f)
        orig_filename = os.path.splitext(orig_filename)[0]
        filename = "data/bert_score_aya_dora/" + orig_filename + "_scores.txt"
        
        score_df = pd.DataFrame()
        score_df["P"] = p
        score_df["R"] = r
        score_df["F1"] = f1

        score_txt = []
        score_txt.append("Mean Precision for {}: {:.4f}".format(orig_filename, score_df['P'].mean()))
        score_txt.append("Max Precision for {}: {:.4f}".format(orig_filename, score_df['P'].max()))
        score_txt.append("Min Precision for {}: {:.4f}".format(orig_filename, score_df['P'].min()))
        score_txt.append("Mean Recall for {}: {:.4f}".format(orig_filename, score_df['R'].mean()))
        score_txt.append("Max Recall for {}: {:.4f}".format(orig_filename, score_df['R'].max()))
        score_txt.append("Min Recall for {}: {:.4f}".format(orig_filename, score_df['R'].min()))
        score_txt.append("Mean F1 for {}: {:.4f}".format(orig_filename, score_df['F1'].mean()))
        score_txt.append("Max F1 for {}: {:.4f}".format(orig_filename, score_df['F1'].max()))
        score_txt.append("Min F1 for {}: {:.4f}".format(orig_filename, score_df['F1'].min()))

        with open(filename, "w") as txt_file:
            for line in score_txt:
                txt_file.write(line + "\n")