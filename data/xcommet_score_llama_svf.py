import os
import re

import numpy as np
import torch
from comet import download_model, load_from_checkpoint


def load_lines(path, strip_special=False):
    with open(path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle.readlines()]
    if strip_special:
        lines = [re.sub(r"<pad>", "", line) for line in lines]
        lines = [re.sub(r"</s>", "", line) for line in lines]
        lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    return lines


def calc_xcomet(model, sources, translations, references):
    data = []
    for src, mt, ref in zip(sources, translations, references):
        data.append({"src": src, "mt": mt, "ref": ref})
    gpus = 1 if torch.cuda.is_available() else 0
    model_output = model.predict(data, batch_size=8, gpus=gpus)
    return model_output


if __name__ == "__main__":
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)

    input_dir_str = "data/encoded_limited_lines_aya/"
    pred_dir_str = "data/llama31_pred_partial_svf/"
    gold_dir_str = "data/parsed/"

    pred_dir = sorted(os.listdir(pred_dir_str))

    for filename in pred_dir:
        src_lines = load_lines(os.path.join(input_dir_str, filename), strip_special=True)
        pred_lines = load_lines(os.path.join(pred_dir_str, filename))
        ref_lines = load_lines(os.path.join(gold_dir_str, filename))

        if not (len(src_lines) == len(pred_lines) == len(ref_lines)):
            raise ValueError(
                f"Mismatched line counts in {filename}: "
                f"src={len(src_lines)}, pred={len(pred_lines)}, ref={len(ref_lines)}"
            )

        model_output = calc_xcomet(model, src_lines, pred_lines, ref_lines)
        avg_score = float(np.mean(model_output.scores))

        base = os.path.splitext(filename)[0]
        out_dir = "data/xcommet_score_llama_svf/"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, base + "_scores.txt"), "w") as txt_file:
            txt_file.write(f"model output score for {base}: {avg_score}\n")

