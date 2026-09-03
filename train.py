import argparse
import json
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from model.daar import build_model
from preprocessing.prepare_model_input import compute_k_max, encode_ids, preprocess


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path):
    config = load_config(config_path)

    with open(config["data"]["path"], "r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    required_columns = [
        config["data"]["user_column"],
        config["data"]["item_column"],
        config["data"]["rating_column"],
        config["data"]["aspect_column"],
        config["data"]["embedding_column"],
        config["data"]["sentiment_column"],
    ]
    df = df[required_columns].copy()
    df.columns = ["user_id", "asin", "rating", "aspects", "embeddings", "sentiments"]

    df, _, _ = encode_ids(df)

    k_max = compute_k_max(df, config["preprocessing"]["k_percentile"])
    print(f"K_max (75th percentile): {k_max}")

    train_df, test_df = train_test_split(
        df,
        test_size=config["preprocessing"]["test_size"],
        random_state=config["preprocessing"]["random_state"],
    )

    embedding_dim = config["preprocessing"]["aspect_embedding_dim"]
    x_train, y_train = preprocess(train_df, k_max, embedding_dim)
    x_test, y_test = preprocess(test_df, k_max, embedding_dim)

    model = build_model(
        num_users=df["user_id"].nunique(),
        num_items=df["asin"].nunique(),
        k_max=k_max,
        sentiment_dim=embedding_dim,
        embedding_dim=config["model"]["user_item_embedding_dim"],
        learning_rate=config["model"]["learning_rate"],
        num_heads=config["model"]["num_attention_heads"],
        key_dim=config["model"]["attention_key_dim"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=config["training"]["early_stopping_min_delta"],
        patience=config["training"]["early_stopping_patience"],
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        validation_split=config["preprocessing"]["validation_split"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=[early_stopping],
    )

    model_path = config["training"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Save the held-out test tensors for evaluation with the exact same split.
    os.makedirs("artifacts", exist_ok=True)
    import numpy as np

    np.savez_compressed(
        "artifacts/test_data.npz",
        user_ids=x_test[0],
        item_ids=x_test[1],
        aspect_embeddings=x_test[2],
        sentiment_probs=x_test[3],
        aspect_mask=x_test[4],
        ratings=y_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
