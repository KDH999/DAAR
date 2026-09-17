import argparse
import os

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm


def clean_aspects(row, nlp, text_column="text"):
    """Filter extracted aspect terms using the rules in the experiment notebook."""
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

        # Single-word terms are retained only when spaCy identifies a noun.
        if " " not in aspect_lower:
            doc = nlp(aspect_lower)
            if not any(token.pos_ == "NOUN" for token in doc):
                continue

        valid.append(aspect)

    return valid


def core_filtering_once(df, user_column, item_column, min_count=5):
    """Apply the user-then-item filtering used in the supplied notebook."""
    user_counts = df[user_column].value_counts()
    valid_users = user_counts[user_counts >= min_count].index
    df = df[df[user_column].isin(valid_users)]

    item_counts = df[item_column].value_counts()
    valid_items = item_counts[item_counts >= min_count].index
    return df[df[item_column].isin(valid_items)]


def main(args):
    nlp = spacy.load(args.spacy_model)
    df = pd.read_json(args.input, lines=args.lines)

    required = [args.user_column, args.item_column, args.rating_column, args.text_column, "aspects"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tqdm.pandas(desc="Postprocessing aspects")
    df["aspects"] = df.progress_apply(
        lambda row: clean_aspects(row, nlp, args.text_column),
        axis=1,
    )

    # Reviews with no valid aspect terms are removed after ATE postprocessing.
    df = df[df["aspects"].map(len) > 0].reset_index(drop=True)

    # The supplied Baby notebook first applies 5-core filtering, then removes
    # duplicate review records, and finally reapplies 5-core filtering because
    # duplicate removal can reduce user/item interaction counts.
    if args.five_core:
        df = core_filtering_once(
            df,
            user_column=args.user_column,
            item_column=args.item_column,
            min_count=args.min_count,
        ).reset_index(drop=True)

    if args.drop_duplicates:
        subset = [
            args.user_column,
            args.item_column,
            args.rating_column,
            args.text_column,
        ]
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

        if args.five_core:
            df = core_filtering_once(
                df,
                user_column=args.user_column,
                item_column=args.item_column,
                min_count=args.min_count,
            ).reset_index(drop=True)

    aspect_counts = df["aspects"].map(len).to_numpy()
    if len(aspect_counts) == 0:
        raise ValueError("No reviews remain after postprocessing.")

    if args.k_max is None:
        percentile_value = np.percentile(aspect_counts, args.percentile)
        k_max = max(1, int(np.ceil(percentile_value)))
    else:
        percentile_value = None
        k_max = args.k_max

    # Truncation is completed before Phrase-BERT and DeBERTa inference.
    df["aspects"] = df["aspects"].apply(lambda values: values[:k_max])

    print(f"Rows after postprocessing: {len(df):,}")
    if percentile_value is not None:
        print(f"{args.percentile:g}th percentile: {percentile_value:.4f}")
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
    parser.add_argument("--user-column", default="user_id")
    parser.add_argument("--item-column", default="asin")
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--percentile", type=float, default=75.0)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--five-core", action="store_true")
    parser.add_argument("--drop-duplicates", action="store_true")
    parser.add_argument("--lines", action="store_true")
    main(parser.parse_args())
