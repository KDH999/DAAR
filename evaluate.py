import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def main(test_path, model_path):
    data = np.load(test_path)

    x_test = [
        data["user_ids"],
        data["item_ids"],
        data["aspect_embeddings"],
        data["sentiment_probs"],
        data["aspect_mask"],
    ]
    y_test = data["ratings"]

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(x_test).reshape(-1)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="artifacts/test_data.npz")
    parser.add_argument("--model", default="checkpoints/daar.keras")
    args = parser.parse_args()
    main(args.test_data, args.model)
