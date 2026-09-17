import argparse
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main(args):
    df = pd.read_json(args.input, lines=args.lines)
    model = SentenceTransformer(args.model_name)

    tqdm.pandas(desc="Embedding aspect terms")

    def encode(aspects):
        if not aspects:
            return []
        return model.encode(aspects, show_progress_bar=False).tolist()

    df["embeddings"] = df["aspects"].progress_apply(encode)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_json(args.output, orient="records", indent=2, force_ascii=False)
    print(f"Saved {len(df):,} records to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="whaleloops/phrase-bert")
    parser.add_argument("--lines", action="store_true")
    main(parser.parse_args())
