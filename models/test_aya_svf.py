# pip install -q transformers peft accelerate torch
import os
import time
import torch
import transformers
from peft import PeftModel
import sys

def main(svf_path):
    start = time.time()

    # Base Aya model
    model_id = "CohereForAI/aya-expanse-8b"
    print(f"Loading base model: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Load base model
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load SVF adapter  
    print(f"Loading SVF adapter from: {svf_path}")
    model = PeftModel.from_pretrained(base_model, svf_path) 
    model = model.to("cuda")   #Move the full model to GPU
    model.eval()    

    # Directory of encoded/masked text input files
    input_dir = "data/encoded_limited_lines_aya/"
    output_dir = "data/aya_pred_partial_svf/"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files
    for in_file in os.scandir(input_dir):
        if not in_file.is_file():
            continue

        print(f"\nProcessing file: {in_file.name}")
        with open(in_file.path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            n_grams = [line for line in text.split('\n') if line.strip()]

        prompt_prefix = (
            "You are a helpful assistant for predicting masked words based on context. "
            "Please return the sentence with your prediction replacing the masked word, "
            "and do not include any explanation or extra text: "
        )

        decoded_outs = []

        for i, x in enumerate(n_grams, start=1):
            chat = [{'role': 'user', 'content': prompt_prefix + x}]

            # Tokenize input in Aya chat format
            input_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            # Generate model response
            with torch.no_grad():
                gen_tokens = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.3,
                )

            gen_text = tokenizer.decode(gen_tokens[0])

            # Parse Aya chat response format
            start_token = "<|CHATBOT_TOKEN|>"
            end_token = "<|END_OF_TURN_TOKEN|>"
            start_idx = gen_text.find(start_token) + len(start_token)
            end_idx = gen_text.find(end_token, start_idx)
            clean_text = gen_text[start_idx:end_idx].strip()

            # Remove markdown and ignore unwanted lines
            clean_text = clean_text.replace('**', '')
            if (
                clean_text
                and not clean_text.startswith('(')
                and not clean_text.startswith('The predicted word is')
            ):
                decoded_outs.append(clean_text)

            print(f"  Line {i}/{len(n_grams)} done")

        # Save predictions
        orig_filename = os.path.splitext(os.path.basename(in_file.name))[0]
        output_path = os.path.join(output_dir, orig_filename + ".txt")

        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(decoded_outs))

        print(f"Saved decoded predictions to {output_path}")

    end = time.time()
    print("\n All files processed.")
    print(" Total time: ", round(end - start, 2), "seconds")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_dora_decode.py <dora_adapter_path>")
        sys.exit(1)

    svf_path=sys.argv[1]
    main(svf_path)