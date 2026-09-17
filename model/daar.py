import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    MultiHeadAttention,
    Multiply,
    Permute,
    RepeatVector,
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
    """Build the DAAR rating-prediction model."""

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
    aspect_mask_input = Input(
        shape=(k_max,),
        dtype=tf.float32,
        name="aspect_mask",
    )

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

    # Mask both padded query and key positions during self-attention.
    key_mask = RepeatVector(k_max, name="key_mask_repeat")(aspect_mask_input)
    query_mask = Permute((2, 1), name="query_mask_transpose")(key_mask)
    attention_mask = Multiply(name="attention_mask")([key_mask, query_mask])

    attention_output = MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=attention_key_dim,
        name="multihead_attention",
    )(
        query=weighted_aspects,
        value=weighted_aspects,
        key=weighted_aspects,
        attention_mask=attention_mask,
    )

    # Keep padded query positions at zero before flattening.
    output_mask = RepeatVector(
        aspect_embedding_dim,
        name="output_mask_repeat",
    )(aspect_mask_input)
    output_mask = Permute((2, 1), name="output_mask_transpose")(output_mask)
    attention_output = Multiply(name="masked_attention_output")(
        [attention_output, output_mask]
    )

    aspect_features = Flatten(name="flatten_attention_output")(attention_output)
    aspect_features = Dense(
        1024,
        activation="relu",
        name="flat_dense2",
    )(aspect_features)

    features = Concatenate(name="final_concat")([user_item, aspect_features])
    features = Dense(128, activation="relu", name="final_dense1")(features)
    features = Dense(32, activation="relu", name="final_dense2")(features)
    output = Dense(1, activation="linear", name="output")(features)

    model = Model(
        inputs=[
            user_input,
            item_input,
            aspect_input,
            sentiment_input,
            aspect_mask_input,
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
