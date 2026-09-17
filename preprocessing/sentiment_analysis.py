import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(args):
    df = pd.read_json(args.input, lines=args.lines)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def classify(row):
        aspects = row.get("aspects", [])
        sentence = row.get(args.text_column, "")
        if not aspects or not isinstance(sentence, str):
            return []

        sentiment_probs = []
        for aspect in aspects:
            prompt = f"{aspect} [SEP] {sentence}"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1).squeeze().tolist()

            sentiment_probs.append([round(float(p), 4) for p in probs])

        return sentiment_probs

    tqdm.pandas(desc="Aspect-level sentiment inference")
    df["sentiments"] = df.progress_apply(classify, axis=1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_json(args.output, orient="records", indent=2, force_ascii=False)
    print(f"Saved {len(df):,} records to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--model-name",
        default="yangheng/deberta-v3-base-absa-v1.1",
    )
    parser.add_argument("--lines", action="store_true")
    main(parser.parse_args())
