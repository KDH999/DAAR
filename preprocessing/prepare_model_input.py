import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_ids(df):
    """Encode user and item identifiers as consecutive integer indices."""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df = df.copy()
    df["user_id"] = user_encoder.fit_transform(df["user_id"])
    df["asin"] = item_encoder.fit_transform(df["asin"])
    return df, user_encoder, item_encoder


def preprocess(df, k_max, embedding_dim=768):
    """Convert a dataframe into fixed-size DAAR model inputs.

    Aspect and sentiment sequences longer than ``k_max`` are truncated.
    Shorter sequences are zero-padded so every sample in a batch has the same
    tensor shape. The original experimental implementation did not apply an
    attention mask to padded positions.
    """
    user_ids = np.asarray(df["user_id"].tolist(), dtype=np.int32)
    item_ids = np.asarray(df["asin"].tolist(), dtype=np.int32)
    ratings = np.asarray(df["rating"].tolist(), dtype=np.float32)

    aspect_embeddings = np.zeros(
        (len(df), k_max, embedding_dim),
        dtype=np.float32,
    )
    sentiment_probs = np.zeros((len(df), k_max, 3), dtype=np.float32)

    for i, (embeddings, sentiments) in enumerate(
        zip(df["embeddings"], df["sentiments"])
    ):
        n = min(len(embeddings), len(sentiments), k_max)
        if n == 0:
            continue

        aspect_embeddings[i, :n] = np.asarray(
            embeddings[:n], dtype=np.float32
        )
        sentiment_probs[i, :n] = np.asarray(
            sentiments[:n], dtype=np.float32
        )

    inputs = [
        user_ids,
        item_ids,
        aspect_embeddings,
        sentiment_probs,
    ]
    return inputs, ratings
