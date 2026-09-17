import argparse
import os

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm


def clean_aspects(row, nlp, text_column="text"):
    text = row[text_column]
    aspects = row.get("aspects", [])

    if not isinstance(text, str) or not isinstance(aspects, list):
        return []

    text_lower = text.lower()
    valid = []

    for aspect in aspects:
        if not isinstance(aspect, str):
            continue

        aspect_lower = aspect.lower().strip()
        if not aspect_lower or aspect_lower not in text_lower:
            continue

        # In the original notebook, single-word terms were retained only when
        # spaCy tagged the term as a noun.
        if " " not in aspect_lower:
            doc = nlp(aspect_lower)
            if not any(token.pos_ == "NOUN" for token in doc):
                continue

        valid.append(aspect)

    return valid


def core_filtering_once(df, min_count=5):
    """Apply the same one-pass user-then-item 5-core filtering as the notebook."""
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_count].index
    df = df[df["user_id"].isin(valid_users)]

    item_counts = df["asin"].value_counts()
    valid_items = item_counts[item_counts >= min_count].index
    return df[df["asin"].isin(valid_items)]


def main(args):
    nlp = spacy.load(args.spacy_model)
    df = pd.read_json(args.input, lines=args.lines)

    tqdm.pandas(desc="Postprocessing aspects")
    df["aspects"] = df.progress_apply(
        lambda row: clean_aspects(row, nlp, args.text_column),
        axis=1,
    )

    # Reviews with no valid aspect terms were removed in the notebook.
    df = df[df["aspects"].map(len) > 0].reset_index(drop=True)

    if args.drop_duplicates:
        subset = ["user_id", "asin", "rating", args.text_column]
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    if args.five_core:
        df = core_filtering_once(df, min_count=args.min_count).reset_index(drop=True)

    aspect_counts = df["aspects"].apply(len).to_numpy()
    if args.k_max is None:
        k_max = max(1, int(np.ceil(np.percentile(aspect_counts, args.percentile))))
    else:
        k_max = args.k_max

    df["aspects"] = df["aspects"].apply(lambda values: values[:k_max])

    print(f"Rows after postprocessing: {len(df):,}")
    print(f"K_max: {k_max}")
    print(f"Mean aspect count before truncation: {aspect_counts.mean():.2f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_json(args.output, orient="records", indent=2, force_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--percentile", type=float, default=75.0)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--five-core", action="store_true")
    parser.add_argument("--drop-duplicates", action="store_true")
    parser.add_argument("--lines", action="store_true")
    main(parser.parse_args())
