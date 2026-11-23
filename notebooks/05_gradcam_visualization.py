
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/best_custom_cnn.keras")
TEST_DIR = Path("data/raw/chest_xray/test")
OUTPUT_DIR = Path("outputs/gradcam_examples")

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGES_PER_CLASS = 1

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. CHECK PATHS
# ============================================================

print("\nChecking project paths...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Build model if necessary.
dummy_input = tf.zeros(
    (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    dtype=tf.float32
)

_ = model(dummy_input, training=False)

print("\nModel architecture:")
model.summary()


# ============================================================
# 4. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError(
        "No Conv2D layer found in the model."
    )

print(
    "\nLast convolutional layer:",
    last_conv_layer.name
)


# ============================================================
# 5. CREATE FEATURE EXTRACTOR
# ============================================================

# Build a model that returns the output of the final
# convolutional layer and the final model prediction.
#
# This implementation assumes a Sequential model with
# the Conv2D layer directly accessible in model.layers.

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=last_conv_layer.output
)

# Layers after the last convolutional layer.
last_conv_index = model.layers.index(last_conv_layer)

post_conv_layers = model.layers[last_conv_index + 1:]


def predict_from_features(feature_maps, training=False):
    """
    Pass convolutional feature maps through all remaining
    layers to obtain the model prediction.
    """

    x = feature_maps

    for layer in post_conv_layers:
        x = layer(x, training=training)

    return x


print("Feature extractor created successfully!")


# ============================================================
# 6. LOAD AND PREPROCESS IMAGE
# ============================================================

def load_image(img_path):

    img = image.load_img(
        img_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    img_array = image.img_to_array(img)

    # Normalize pixels to 0-1.
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# 7. GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(img_array):

    img_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )

    # Get convolutional feature maps.
    conv_outputs = feature_extractor(
        img_tensor,
        training=False
    )

    with tf.GradientTape() as tape:

        # Watch the feature maps directly.
        tape.watch(conv_outputs)

        # Continue forward pass from convolutional layer.
        predictions = predict_from_features(
            conv_outputs,
            training=False
        )

        # Binary classification with sigmoid output.
        class_score = predictions[:, 0]

    # Calculate gradients with respect to feature maps.
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are still None. "
            "Please check the model architecture and output layers."
        )

    # Average gradients across spatial dimensions.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Get feature maps and weights for first image.
    feature_maps = conv_outputs[0]
    weights = pooled_gradients[0]

    # Weighted combination of feature maps.
    heatmap = tf.reduce_sum(
        feature_maps * weights,
        axis=-1
    )

    # ReLU.
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap.
    heatmap = tf.math.divide_no_nan(
        heatmap,
        tf.reduce_max(heatmap)
    )

    prediction_score = float(
        predictions[0][0].numpy()
    )

    return heatmap.numpy(), prediction_score


# ============================================================
# 8. SAVE GRAD-CAM VISUALIZATION
# ============================================================

def save_gradcam(img_path, actual_class, image_number):

    print(f"\nProcessing: {img_path.name}")

    img_array = load_image(img_path)

    heatmap, prediction_score = generate_gradcam(
        img_array
    )

    # Load original image.
    original_img = image.load_img(
        img_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    original_img = image.img_to_array(
        original_img
    ) / 255.0

    # Resize heatmap.
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMAGE_SIZE
    ).numpy().squeeze()

    # Create colored heatmap.
    colored_heatmap = plt.get_cmap("jet")(
        heatmap_resized
    )[..., :3]

    # Overlay heatmap.
    overlay = (
        0.6 * original_img +
        0.4 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # Binary classification threshold.
    predicted_class = (
        "PNEUMONIA"
        if prediction_score >= 0.5
        else "NORMAL"
    )

    print("Actual class:", actual_class)
    print("Predicted class:", predicted_class)
    print(
        "Pneumonia score:",
        round(prediction_score, 4)
    )

    # Create figure.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(original_img)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap_resized,
        cmap="jet"
    )
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {predicted_class}"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Actual Class: {actual_class}",
        fontsize=14
    )

    plt.tight_layout()

    # Save output.
    output_path = OUTPUT_DIR / (
        f"{actual_class.lower()}_gradcam_{image_number}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# 9. PROCESS TEST IMAGES
# ============================================================

print("\nStarting Grad-CAM visualization...")

for class_name in CLASS_NAMES:

    class_dir = TEST_DIR / class_name

    if not class_dir.exists():
        print(
            f"Warning: Folder not found: {class_dir}"
        )
        continue

    image_files = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in (
            ".jpg",
            ".jpeg",
            ".png"
        )
    ]

    if not image_files:
        print(
            f"Warning: No images found in {class_dir}"
        )
        continue

    selected_images = random.sample(
        image_files,
        min(
            IMAGES_PER_CLASS,
            len(image_files)
        )
    )

    for index, img_path in enumerate(
        selected_images,
        start=1
    ):

        save_gradcam(
            img_path,
            class_name,
            index
        )


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n===================================")
print("GRAD-CAM VISUALIZATION COMPLETED!")
print("===================================")

print(
    "\nOutput folder:",
    OUTPUT_DIR.resolve()
)

print(
    "\nOpen outputs/gradcam_examples "
    "to view your generated images."
)
