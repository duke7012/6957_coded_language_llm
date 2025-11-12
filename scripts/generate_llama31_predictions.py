#!/usr/bin/env python
"""
Generate translation predictions with a fine-tuned Llama 3.1 adapter.

The script mirrors the formatting used during fine-tuning so that the
resulting text files can be consumed by the evaluation utilities
(`data/bert_score_aya_*.py`, `data/mover_score_aya_*.py`, etc.).
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a translator that converts encoded or foreign text into plain English. "
    "When given input text, translate it accurately to English."
)


def load_model(
    adapter_path: str,
    base_model_name: str,
    use_gpu: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """Load the base model, apply the adapter, and prepare the tokenizer."""
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, device


def read_inputs(path: Path) -> List[str]:
    """Read non-empty lines from a text file."""
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]
    return [line for line in lines if line]


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def generate_translation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    source_text: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    """Generate a translation for a single input string."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate this text to English: {source_text}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


def iterate_input_files(input_dir: Path, only_files: Iterable[str] | None) -> List[Path]:
    files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if only_files is not None:
        wanted = set(only_files)
        files = [p for p in files if p.name in wanted]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fine-tuned Llama 3.1 adapter to generate translation outputs."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Directory containing the saved PEFT adapter weights (e.g. dora_results_llama31_8b/epochs_3/).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model to load before applying the adapter.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/encoded_limited_lines_aya",
        help="Directory with source text files to translate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/llama31_pred_partial",
        help="Directory where translated files will be written.",
    )
    parser.add_argument(
        "--only-files",
        type=str,
        nargs="*",
        help="Optional list of specific filenames (within input-dir) to process.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt passed to the chat template.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (0 disables).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model, tokenizer, device = load_model(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        use_gpu=not args.no_gpu,
    )

    files_to_process = iterate_input_files(input_dir, args.only_files)

    if not files_to_process:
        raise RuntimeError("No input files found to process.")

    for input_file in tqdm(files_to_process, desc="Files"):
        source_lines = read_inputs(input_file)
        predictions: List[str] = []

        for line in tqdm(source_lines, desc=input_file.name, leave=False):
            translation = generate_translation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                source_text=line,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            predictions.append(translation)

        relative_path = input_file.relative_to(input_dir)
        output_path = output_dir / relative_path
        ensure_output_dir(output_path)

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(predictions))


if __name__ == "__main__":
    main()

