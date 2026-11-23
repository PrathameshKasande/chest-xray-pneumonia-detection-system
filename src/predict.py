from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_custom_cnn.keras"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src/train.py first."
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    model = tf.keras.models.load_model(MODEL_PATH)

    score = float(model.predict(image_array, verbose=0)[0][0])
    predicted_index = int(score >= 0.5)

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "model_score": score,
        "normal_score": 1.0 - score,
        "pneumonia_score": score
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path_to_xray_image")
    else:
        result = predict_image(sys.argv[1])
        print(result)