import numpy as np
import os
import re

from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)

class ScoreObj:
    def __init__(self, score, system_score, error_span):
        self.score = score
        self.system_score = system_score
        self.error_span = error_span

def calc_xcommet(inputs, outputs, golds):

    data = []

    for i, o, g in zip(inputs, outputs, golds):
        data_obj = {"src": i, "mt": o, "ref": g}
        data.append(data_obj)
        
    scores = []
    
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    score_obj = ScoreObj(model_output.scores, model_output.system_score, model_output.metadata.error_spans)
    
    scores.append(score_obj)
    
    return scores



if __name__ == '__main__':
    # need input file
    input_dir_str = "data/encoded_limited_lines_aya/"
    input_dir = sorted(os.listdir(input_dir_str))
    
    # need prediction file
    # get dir for aya
    pred_dir_str = "data/aya_pred_partial_dora/"
    pred_dir = sorted(os.listdir(pred_dir_str))
    
    # need parsed file
    gold_dir_str = "data/parsed/"
    gold_dir = sorted(os.listdir(gold_dir_str))
    
    
    for pred_file in pred_dir:
        in_file = pred_file
        gold_file = pred_file
        
        with open(pred_dir_str + pred_file, 'r') as file:
            pred_lines = file.readlines()
            # Remove trailing newline characters from each line
            pred_lines = [line.rstrip('\n') for line in pred_lines]
            pred_lines = [string for string in pred_lines if string]
            

        with open(input_dir_str + in_file, 'r') as file:
            gold_lines = file.readlines()
            # Remove trailing newline characters from each line
            gold_lines = [line.rstrip('\n') for line in gold_lines]
            gold_lines = [re.sub(r"<pad>", "", line) for line in gold_lines]
            gold_lines = [re.sub(r"</s>", "", line) for line in gold_lines]
            gold_lines = [line.strip() for line in gold_lines]
            gold_lines = [string for string in gold_lines if string]
        
        with open(gold_dir_str + gold_file, 'r') as file:
            gold_lines = file.readlines()

        scores = calc_xcommet(pred_lines, gold_lines, gold_lines)
        
        orig_filename = os.path.basename(gold_file)
        orig_filename = os.path.splitext(orig_filename)[0]
        filename = "data/xcommet_score_aya_dora/" + orig_filename + "_scores.txt"
        
        scores_mean = np.mean(np.array(scores[0].score))
        
        score_txt = []
        score_txt.append("model output score for {}: {}".format(orig_filename, scores_mean))
        
        with open(filename, "w") as txt_file:
            for line in score_txt:
                txt_file.write(line + "\n")
        
        print("xcommet system score average for {}: {}".format(orig_filename, scores_mean))