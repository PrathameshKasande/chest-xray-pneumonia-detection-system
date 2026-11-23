from pathlib import Path
import pandas as pd

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_LABEL = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_dataset_root(project_root):
    raw_dir = Path(project_root) / "data" / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")

    candidates = [raw_dir] + [
        folder for folder in raw_dir.rglob("*")
        if folder.is_dir()
    ]

    for folder in candidates:
        if all((folder / split).is_dir()
               for split in ["train", "test"]):
            return folder

    raise FileNotFoundError(
        "Dataset not found. Expected train and test folders "
        "with NORMAL and PNEUMONIA subfolders."
    )


def collect_images(folder, split_name):
    folder = Path(folder)
    records = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = folder / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "filepath": str(image_path.resolve()),
                    "class_name": class_name,
                    "label": label,
                    "split": split_name
                })

    if not records:
        raise ValueError(f"No images found in {folder}")

    return pd.DataFrame(records)


def load_dataframes(project_root):
    dataset_root = find_dataset_root(project_root)

    train_df = collect_images(dataset_root / "train", "train")
    test_df = collect_images(dataset_root / "test", "test")

    # We create a new validation split from development data.
    # This avoids depending on the original tiny validation folder.
    val_dir = dataset_root / "val"

    if val_dir.exists():
        original_val_df = collect_images(val_dir, "original_val")
        development_df = pd.concat(
            [train_df, original_val_df],
            ignore_index=True
        )
    else:
        development_df = train_df.copy()

    return development_df, test_df, dataset_root