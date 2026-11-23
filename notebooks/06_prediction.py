
from pathlib import Path
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

# Configuration
MODEL_PATH = Path("models/best_custom_cnn.keras")
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def predict_xray(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Normalize image
    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # Add batch dimension
    input_image = np.expand_dims(image_array, axis=0)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Predict
    probability = float(
        model.predict(input_image, verbose=0)[0][0]
    )

    predicted_class = (
        "PNEUMONIA" if probability >= 0.5
        else "NORMAL"
    )

    print("\nCHEST X-RAY PREDICTION")
    print("-----------------------------")
    print("Predicted class:", predicted_class)
    print(f"Pneumonia model score: {probability:.4f}")
    print(f"Normal model score: {1 - probability:.4f}")

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Model prediction: {predicted_class}")
    plt.axis("off")
    plt.show()

    print("\nEducational research output only.")
    print("This model is not a medical diagnostic tool.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict a chest X-ray image"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the chest X-ray image"
    )

    args = parser.parse_args()

    predict_xray(args.image)
