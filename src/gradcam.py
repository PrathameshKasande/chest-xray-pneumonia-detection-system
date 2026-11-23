# Grad-CAM creates a heatmap to help visualize image regions that influenced the CNN output.
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_custom_cnn.keras"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam_examples"

IMAGE_SIZE = (224, 224)


def make_gradcam_heatmap(image_array, model):
    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No convolutional layer found.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def save_gradcam(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image) / 255.0
    batch = np.expand_dims(image_array, axis=0)

    heatmap = make_gradcam_heatmap(batch, model)

    heatmap = cv2.resize(
        heatmap,
        (IMAGE_SIZE[1], IMAGE_SIZE[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original = np.uint8(image_array * 255)
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "gradcam_result.jpg"
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM visualization")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Saved:", output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/gradcam.py path_to_xray_image")
    else:
        save_gradcam(sys.argv[1])