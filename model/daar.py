import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    MultiHeadAttention,
    Multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(
    num_users,
    num_items,
    k_max,
    aspect_embedding_dim=768,
    user_item_embedding_dim=128,
    sentiment_hidden_dim=64,
    num_attention_heads=6,
    attention_key_dim=128,
    learning_rate=1e-4,
):
    """Build the DAAR rating-prediction model.

    The implementation follows the TensorFlow experiment notebook used for the
    paper. Aspect sequences are zero-padded to ``k_max`` before entering the
    model. Padding positions are not masked inside the attention layer, which
    matches the original experimental implementation.
    """

    user_input = Input(shape=(), dtype=tf.int32, name="user_input")
    item_input = Input(shape=(), dtype=tf.int32, name="item_input")
    aspect_input = Input(
        shape=(k_max, aspect_embedding_dim),
        dtype=tf.float32,
        name="aspect_input",
    )
    sentiment_input = Input(
        shape=(k_max, 3),
        dtype=tf.float32,
        name="sentiment_input",
    )

    # User-item interaction representation
    user_embedding = Embedding(
        num_users,
        user_item_embedding_dim,
        name="user_embedding",
    )(user_input)
    item_embedding = Embedding(
        num_items,
        user_item_embedding_dim,
        name="item_embedding",
    )(item_input)
    user_item = Concatenate(name="user_item_concat")(
        [user_embedding, item_embedding]
    )
    user_item = Dense(128, activation="relu", name="user_item_mlp")(user_item)

    # Continuous sentiment reflection
    sentiment_features = Dense(
        sentiment_hidden_dim,
        name="sentiment_dense1",
    )(sentiment_input)
    sentiment_features = Dense(
        aspect_embedding_dim,
        name="sentiment_dense3",
    )(sentiment_features)
    weighted_aspects = Multiply(name="weighted_aspects")(
        [aspect_input, sentiment_features]
    )

    # Multi-head self-attention over sentiment-aware aspect representations
    attention_output = MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=attention_key_dim,
        name="multihead_attention",
    )(
        query=weighted_aspects,
        value=weighted_aspects,
        key=weighted_aspects,
    )

    aspect_features = Flatten(name="flatten_attention_output")(attention_output)
    aspect_features = Dense(
        1024,
        activation="relu",
        name="flat_dense2",
    )(aspect_features)

    # Final rating prediction
    features = Concatenate(name="final_concat")([user_item, aspect_features])
    features = Dense(128, activation="relu", name="final_dense1")(features)
    features = Dense(32, activation="relu", name="final_dense2")(features)
    output = Dense(1, activation="linear", name="output")(features)

    model = Model(
        inputs=[user_input, item_input, aspect_input, sentiment_input],
        outputs=output,
        name="DAAR",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model
