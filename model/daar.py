import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Multiply,
    Concatenate,
    Flatten,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(
    num_users,
    num_items,
    k_max,
    sentiment_dim=768,
    embedding_dim=256,
    learning_rate=1e-4,
    num_heads=6,
    key_dim=128,
):
    """Build the DAAR rating prediction model."""

    user_input = Input(shape=(), dtype=tf.int32, name="user_input")
    item_input = Input(shape=(), dtype=tf.int32, name="item_input")
    aspect_input = Input(
        shape=(k_max, sentiment_dim), dtype=tf.float32, name="aspect_input"
    )
    sentiment_input = Input(
        shape=(k_max, 3), dtype=tf.float32, name="sentiment_input"
    )
    aspect_mask = Input(shape=(k_max,), dtype=tf.bool, name="aspect_mask")

    # User-item interaction representation
    user_embed = Embedding(num_users, embedding_dim, name="user_embedding")(user_input)
    item_embed = Embedding(num_items, embedding_dim, name="item_embedding")(item_input)
    uv_concat = Concatenate(name="user_item_concat")([user_embed, item_embed])
    uv_mlp = Dense(128, activation="relu", name="user_item_mlp")(uv_concat)

    # Aspect-level sentiment weighting
    sent_dense1 = Dense(64, name="sentiment_dense1")(sentiment_input)
    sent_dense3 = Dense(sentiment_dim, name="sentiment_dense3")(sent_dense1)
    weighted_aspects = Multiply(name="weighted_aspects")([aspect_input, sent_dense3])

    # Multi-head self-attention over sentiment-aware aspect representations
    attention_mask = tf.keras.layers.Lambda(
        lambda m: tf.logical_and(tf.expand_dims(m, 1), tf.expand_dims(m, 2)),
        name="attention_mask",
    )(aspect_mask)

    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="multihead_attention",
    )(
        query=weighted_aspects,
        value=weighted_aspects,
        key=weighted_aspects,
        attention_mask=attention_mask,
    )

    # Aspect feature transformation
    flat_output = Flatten(name="flatten_attention_output")(attn_output)
    flat_dense2 = Dense(1024, activation="relu", name="flat_dense2")(flat_output)

    # Rating prediction
    final_concat = Concatenate(name="final_concat")([uv_mlp, flat_dense2])
    final_dense1 = Dense(128, activation="relu", name="final_dense1")(final_concat)
    final_dense2 = Dense(32, activation="relu", name="final_dense2")(final_dense1)
    output = Dense(1, activation="linear", name="output")(final_dense2)

    model = Model(
        inputs=[
            user_input,
            item_input,
            aspect_input,
            sentiment_input,
            aspect_mask,
        ],
        outputs=output,
        name="DAAR",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse", "mae"],
    )

    return model
