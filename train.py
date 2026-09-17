import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from model.daar import build_model
from preprocessing.prepare_model_input import encode_ids, preprocess


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
    df.columns = [
        "user_id",
        "asin",
        "rating",
        "aspects",
        "embeddings",
        "sentiments",
    ]

    df, user_encoder, item_encoder = encode_ids(df)

    split_cfg = config["preprocessing"]
    train_df, test_df = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )

    k_max = max(df["aspects"].map(len).max(), 1)
    embedding_dim = split_cfg["aspect_embedding_dim"]
    print(f"Using preprocessed K_max: {k_max}")

    x_train, y_train = preprocess(train_df, k_max, embedding_dim)
    x_test, y_test = preprocess(test_df, k_max, embedding_dim)

    model_cfg = config["model"]
    model = build_model(
        num_users=df["user_id"].nunique(),
        num_items=df["asin"].nunique(),
        k_max=k_max,
        aspect_embedding_dim=embedding_dim,
        user_item_embedding_dim=model_cfg["user_item_embedding_dim"],
        sentiment_hidden_dim=model_cfg["sentiment_hidden_dim"],
        num_attention_heads=model_cfg["num_attention_heads"],
        attention_key_dim=model_cfg["attention_key_dim"],
        learning_rate=model_cfg["learning_rate"],
    )

    training_cfg = config["training"]
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=training_cfg["early_stopping_min_delta"],
        patience=training_cfg["early_stopping_patience"],
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        validation_split=split_cfg["validation_split"],
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        callbacks=[early_stopping],
    )

    model_path = training_cfg["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    artifact_path = training_cfg["test_artifact_path"]
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    np.savez_compressed(
        artifact_path,
        user_ids=x_test[0],
        item_ids=x_test[1],
        aspect_embeddings=x_test[2],
        sentiment_probs=x_test[3],
        aspect_mask=x_test[4],
        ratings=y_test,
    )

    with open("artifacts/id_encoders.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_classes": user_encoder.classes_.tolist(),
                "item_classes": item_encoder.classes_.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
