import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_ids(df):
    """Encode user and item identifiers as integer indices."""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df = df.copy()
    df["user_id"] = user_encoder.fit_transform(df["user_id"])
    df["asin"] = item_encoder.fit_transform(df["asin"])

    return df, user_encoder, item_encoder


def compute_k_max(df, percentile=75):
    """Set K_max from the aspect-count distribution."""
    lengths = df["embeddings"].apply(len).to_numpy()
    if len(lengths) == 0:
        return 1
    return max(1, int(np.ceil(np.percentile(lengths, percentile))))


def preprocess(df, k_max, embedding_dim=768):
    """
    Convert dataframe columns into fixed-size model inputs.

    Aspect sequences longer than K_max are truncated and shorter sequences
    are zero-padded. aspect_mask identifies the valid (non-padding) positions.
    """
    user_ids = np.asarray(df["user_id"].tolist(), dtype=np.int32)
    item_ids = np.asarray(df["asin"].tolist(), dtype=np.int32)
    ratings = np.asarray(df["rating"].tolist(), dtype=np.float32)

    aspect_embeddings = np.zeros(
        (len(df), k_max, embedding_dim), dtype=np.float32
    )
    sentiment_probs = np.zeros((len(df), k_max, 3), dtype=np.float32)
    aspect_mask = np.zeros((len(df), k_max), dtype=np.bool_)

    for i, (embeddings, sentiments) in enumerate(
        zip(df["embeddings"], df["sentiments"])
    ):
        n = min(len(embeddings), len(sentiments), k_max)
        if n == 0:
            continue

        aspect_embeddings[i, :n] = np.asarray(embeddings[:n], dtype=np.float32)
        sentiment_probs[i, :n] = np.asarray(sentiments[:n], dtype=np.float32)
        aspect_mask[i, :n] = True

    inputs = [
        user_ids,
        item_ids,
        aspect_embeddings,
        sentiment_probs,
        aspect_mask,
    ]

    return inputs, ratings
