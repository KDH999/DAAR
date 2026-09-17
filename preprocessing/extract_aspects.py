import argparse
import json
import os
import re
import string

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def generate_prompt(review):
    return (
        "Aspect Term Extraction is the task of identifying aspect terms mentioned in a text. "
        "These aspect terms refer to important attributes or features related to a product or service.\n\n"
        "Your task is as follows:\n"
        "1. Extract only the aspect terms from the given text.\n"
        "2. The output should be a list of the extracted aspect terms in JSON format.\n"
        "3. If there are no aspect terms in the text that meet these conditions, return an empty list.\n\n"
        "Example output format:\n"
        '["aspect term 1", "aspect term 2", "aspect term 3"]\n\n'
        "Please provide the final output based on the following review text in JSON format.\n\n"
        f"Text:\n{review}\n\nResponse:\n"
    )


def is_placeholder(values):
    return bool(values) and all(
        isinstance(x, str) and re.fullmatch(r"aspect term \d+", x)
        for x in values
    )


def parse_aspects(output_text):
    for candidate in re.findall(r"\[.*?\]", output_text, re.DOTALL):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and parsed and not is_placeholder(parsed):
                return [x.strip() for x in parsed if isinstance(x, str) and x.strip()]
        except json.JSONDecodeError:
            pass

    extracted = re.findall(r'"([^"]+)"', output_text)
    if not extracted:
        extracted = re.findall(r"'([^']+)'", output_text)

    if not extracted and "," in output_text:
        cleaned = output_text.replace("[", "").replace("]", "")
        extracted = [x.strip().strip("\"'") for x in cleaned.split(",")]

    seen = set()
    result = []
    for term in extracted:
        term = term.strip()
        if (
            term
            and len(term) < 50
            and term not in seen
            and not re.fullmatch(r"aspect term \d+", term)
        ):
            seen.add(term)
            result.append(term)
    return result


def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_model(model_id):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    return tokenizer, model


def main(args):
    df = pd.read_json(args.input, lines=args.lines)
    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")

    if args.remove_punctuation:
        df[args.text_column] = df[args.text_column].apply(remove_punctuation)

    tokenizer, model = load_model(args.model_id)
    embed_device = model.get_input_embeddings().weight.device

    aspects_all = []
    raw_outputs = []

    for review in tqdm(df[args.text_column].tolist(), desc="Extracting aspects"):
        prompt = generate_prompt(review)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw_outputs.append(output_text)
        aspects_all.append(parse_aspects(output_text))

    df["aspects"] = aspects_all
    if args.keep_raw_output:
        df["ate_raw_output"] = raw_outputs

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_json(args.output, orient="records", indent=2, force_ascii=False)
    print(f"Saved {len(df):,} records to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Checkpoint used in the supplied ATE experiment notebook.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lines", action="store_true")
    parser.add_argument("--remove-punctuation", action="store_true")
    parser.add_argument("--keep-raw-output", action="store_true")
    main(parser.parse_args())
