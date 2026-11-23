
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path("data/raw/chest_xray")
OUTPUT_DIR = Path("processed_data")

IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# 0 = NORMAL, 1 = PNEUMONIA
def load_images(folder):
    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_folder = folder / class_name

        if not class_folder.exists():
            print(f"Warning: {class_folder} does not exist")
            continue

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMAGE_SIZE)

                    array = np.asarray(
                        image,
                        dtype=np.float32
                    ) / 255.0

                    images.append(array)
                    labels.append(label)

            except Exception as error:
                print(f"Skipping {image_path}: {error}")

    return np.array(images), np.array(labels)


# 1. Load original training data
print("Loading training images...")

X_train_all, y_train_all = load_images(DATA_DIR / "train")

print("Training images:", X_train_all.shape)
print("Training labels:", y_train_all.shape)


# 2. Create a stratified validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_all
)


# 3. Load the original test data separately
print("\nLoading test images...")

X_test, y_test = load_images(DATA_DIR / "test")

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)


# 4. Save processed data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "y_val.npy", y_val)

np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)


# 5. Print final summary
print("\nPREPROCESSING COMPLETED")
print("-" * 40)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nClass mapping:")
print("0 = NORMAL")
print("1 = PNEUMONIA")
