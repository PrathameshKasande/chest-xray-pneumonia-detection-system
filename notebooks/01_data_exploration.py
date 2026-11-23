
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

# Dataset location
DATA_DIR = Path("data/raw/chest_xray")

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

# 1. Count images in each split
print("DATASET SUMMARY")
print("-" * 45)

for split in SPLITS:
    print(f"\n{split.upper()}")

    for class_name in CLASSES:
        folder = DATA_DIR / split / class_name

        if not folder.exists():
            print(f"{class_name}: Folder not found")
            continue

        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{class_name}: {len(images)} images")


# 2. Display class distribution
counts = {}

for class_name in CLASSES:
    folder = DATA_DIR / "train" / class_name

    if folder.exists():
        counts[class_name] = len([
            file for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

plt.figure(figsize=(7, 5))
plt.bar(counts.keys(), counts.values())
plt.title("Training Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()


# 3. Display sample images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for row, class_name in enumerate(CLASSES):
    folder = DATA_DIR / "train" / class_name

    if not folder.exists():
        continue

    images = list(folder.glob("*"))[:3]

    for col, image_path in enumerate(images):
        try:
            image = Image.open(image_path).convert("RGB")

            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].set_title(class_name)
            axes[row, col].axis("off")

        except Exception as error:
            print(f"Could not open {image_path}: {error}")

plt.suptitle("Sample Chest X-ray Images")
plt.tight_layout()
plt.show()


# 4. Check image dimensions
dimensions = Counter()

for class_name in CLASSES:
    folder = DATA_DIR / "train" / class_name

    if not folder.exists():
        continue

    for image_path in folder.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            with Image.open(image_path) as image:
                dimensions[image.size] += 1
        except Exception:
            print(f"Invalid image: {image_path}")

print("\nMost common image dimensions:")
for size, count in dimensions.most_common(10):
    print(f"{size}: {count}")
