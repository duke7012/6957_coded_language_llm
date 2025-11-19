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
    "You are a translation engine. Given input text, reply with ONLY the natural "
    "English translation as a single concise sentence. Do not explain, describe "
    "your process, mention decoding, or add any extra commentary. "
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


def extract_translation(generated_text: str) -> str:
    """Extract the actual translation from verbose model output."""
    import re
    
    # Split into lines and filter out empty lines
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    
    # Patterns that indicate explanations (very comprehensive)
    explanation_patterns = [
        r'^it (appears|seems|looks)',
        r'^the (text|encoded|decoded|translation)',
        r'^(decoded|decoding|encoded|encoding)',
        r'^to translate',
        r'^here\'s',
        r'^so the',
        r'^this text',
        r'^using base64',
        r'^now, let\'s',
        r'^i\'ll',
        r'^if you\'d like',
        r'^i\'m (not|happy|able|sure)',
        r'^i think',
        r'^the translation of',
        r'^base64 (encoded|decoding)',
        r'^the (base64|url)',
        r'^after (decoding|analyzing)',
        r'^decoded (string|text|bytes)',
        r'^the encoded',
        r'^it looks like',
        r'^could you please',
        r'^however,',
        r'^first,',
        r'^let me',
        r'^string:',
        r'^text is:',
        r'^text:',
        r'^is encoded',
        r'^is:',
        r'^appears to be',
        r'^seems to be',
        r'^is likely',
        r'^is decoded',
        r'^yields:',
        r'^results in:',
        r'^translates to:',
        r'^decoded to:',
        r'^message is:',
        r'^version:',
    ]
    
    # Base64-like pattern (long alphanumeric strings with = at end)
    base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
    
    def is_explanation_line(line: str) -> bool:
        """Check if a line is an explanation."""
        line_lower = line.lower()
        # Check explanation patterns
        if any(re.match(pat, line_lower) for pat in explanation_patterns):
            return True
        # Check for explanation keywords
        explanation_keywords = [
            'decoded', 'encoded', 'base64', 'translation', 'translate',
            'decoding', 'encoding', 'appears', 'seems', 'looks like',
            'i\'m', 'i think', 'could you', 'let me', 'here\'s'
        ]
        if any(keyword in line_lower for keyword in explanation_keywords):
            # But allow if it's a very short, simple line that might be actual translation
            if len(line.split()) <= 8 and not re.search(base64_pattern, line):
                return False
            return True
        return False
    
    def clean_line(line: str) -> str:
        """Clean a line by removing encoded strings and metadata."""
        # Remove base64-like encoded strings
        line = re.sub(base64_pattern, '', line)
        # Remove common prefixes
        line = re.sub(r'^(decoded|translation|text|string|result|output|message|is|text is|text:)[:\s]+', '', line, flags=re.IGNORECASE)
        # Remove quotes/backticks
        line = re.sub(r'^["\'`]+|["\'`]+$', '', line)
        # Remove URLs
        line = re.sub(r'https?://\S+', '', line)
        # Clean up multiple spaces
        line = re.sub(r'\s+', ' ', line)
        return line.strip()
    
    def looks_like_translation(line: str) -> bool:
        """Check if a line looks like an actual translation."""
        cleaned = clean_line(line)
        if len(cleaned) < 3:
            return False
        # Should have mostly letters and spaces (not too many special chars)
        alpha_ratio = len(re.findall(r'[a-zA-Z]', cleaned)) / max(len(cleaned), 1)
        if alpha_ratio < 0.6:
            return False
        # Should not be mostly numbers or special chars
        if re.match(r'^[\d\s=+/]+$', cleaned):
            return False
        # Should not contain base64 patterns
        if re.search(base64_pattern, cleaned):
            return False
        return True
    
    # First pass: look for lines after colons that might be translations
    for i, line in enumerate(lines):
        # Look for patterns like "text: <translation>" or "decoded: <translation>"
        colon_match = re.search(r'[:\s]+(.+)$', line)
        if colon_match:
            candidate = colon_match.group(1).strip()
            candidate = clean_line(candidate)
            if looks_like_translation(candidate) and not is_explanation_line(candidate):
                return candidate
    
    # Second pass: find lines that look like translations
    candidates = []
    for line in lines:
        if is_explanation_line(line):
            continue
        
        cleaned = clean_line(line)
        if looks_like_translation(cleaned):
            # Prefer shorter, cleaner lines (actual translations are usually concise)
            # But also consider length to avoid picking up fragments
            score = len(cleaned) if 10 <= len(cleaned) <= 200 else 0
            if score > 0:
                candidates.append((score, cleaned))
    
    # Return the best candidate (prefer medium-length, clean translations)
    if candidates:
        # Sort by score (length), prefer medium-length translations
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]
    
    # Third pass: try to extract from any line, even if it has some explanation
    for line in lines:
        cleaned = clean_line(line)
        # Remove explanation prefixes more aggressively
        cleaned = re.sub(r'^(it|the|this|that|here|so|now|first|after|before|when|where|which|who|what|how|why|if|but|and|or|however|therefore|thus|hence|moreover|furthermore|additionally|also|too|as well|in addition|for example|for instance|specifically|namely|that is|i\.e\.|e\.g\.)[,\s]+', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if looks_like_translation(cleaned) and len(cleaned) >= 5:
            return cleaned
    
    # Final fallback: return first line, heavily cleaned
    if lines:
        fallback = clean_line(lines[0])
        # Remove any remaining explanation patterns
        fallback = re.sub(r'^(decoded|translation|text|string|result|output|message|is|text is|text:)[:\s]+', '', fallback, flags=re.IGNORECASE)
        fallback = re.sub(base64_pattern, '', fallback)
        fallback = re.sub(r'\s+', ' ', fallback).strip()
        return fallback if fallback else generated_text.strip()
    
    return generated_text.strip()


def enforce_single_sentence(text: str) -> str:
    """Force the translation to a single concise sentence."""
    import re

    if not text:
        return text

    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return normalized

    # Remove common explanation prefixes
    prefixes = [
        "it appears that",
        "it seems that",
        "the text",
        "decoded text is",
        "translation:",
        "decoded:",
        "result:",
        "message:",
    ]
    lowered = normalized.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].lstrip()
            lowered = normalized.lower()
            break

    # Take the first sentence-ending punctuation if present
    sentence_match = re.search(r"(.+?[.!?])(\s|$)", normalized)
    if sentence_match:
        candidate = sentence_match.group(1).strip()
    else:
        # Otherwise fall back to the first line / chunk
        candidate = normalized.split(".", 1)[0].split("!", 1)[0].split("?", 1)[0].strip()
        if not candidate:
            candidate = normalized

    # Final cleanup
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


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
    
    # Extract clean translation from potentially verbose output
    clean_translation = extract_translation(generated_text)
    clean_translation = enforce_single_sentence(clean_translation)
    return clean_translation


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
        default=0.0,
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

