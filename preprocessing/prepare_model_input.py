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
    """Convert preprocessed DAAR data into fixed-size model inputs."""
    user_ids = np.asarray(df["user_id"].tolist(), dtype=np.int32)
    item_ids = np.asarray(df["asin"].tolist(), dtype=np.int32)
    ratings = np.asarray(df["rating"].tolist(), dtype=np.float32)

    aspect_embeddings = np.zeros(
        (len(df), k_max, embedding_dim),
        dtype=np.float32,
    )
    sentiment_probs = np.zeros((len(df), k_max, 3), dtype=np.float32)
    aspect_mask = np.zeros((len(df), k_max), dtype=np.float32)

    for i, (aspects, embeddings, sentiments) in enumerate(
        zip(df["aspects"], df["embeddings"], df["sentiments"])
    ):
        n = len(aspects)

        if len(embeddings) != n or len(sentiments) != n:
            raise ValueError(
                f"Mismatched aspect features at row {i}: "
                f"aspects={n}, embeddings={len(embeddings)}, sentiments={len(sentiments)}"
            )

        if n > k_max:
            raise ValueError(
                f"Aspect sequence at row {i} exceeds preprocessed K_max={k_max}."
            )

        if n == 0:
            continue

        embedding_array = np.asarray(embeddings, dtype=np.float32)
        sentiment_array = np.asarray(sentiments, dtype=np.float32)

        if embedding_array.shape != (n, embedding_dim):
            raise ValueError(
                f"Unexpected embedding shape at row {i}: {embedding_array.shape}"
            )
        if sentiment_array.shape != (n, 3):
            raise ValueError(
                f"Unexpected sentiment shape at row {i}: {sentiment_array.shape}"
            )

        aspect_embeddings[i, :n] = embedding_array
        sentiment_probs[i, :n] = sentiment_array
        aspect_mask[i, :n] = 1.0

    inputs = [
        user_ids,
        item_ids,
        aspect_embeddings,
        sentiment_probs,
        aspect_mask,
    ]
    return inputs, ratings
