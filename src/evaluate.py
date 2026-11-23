from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from data_utils import load_dataframes
from preprocessing import create_tf_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_custom_cnn.keras"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run src/train.py first."
        )

    _, test_df, _ = load_dataframes(PROJECT_ROOT)
    test_dataset = create_tf_dataset(test_df)

    model = tf.keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(test_dataset).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    actual = test_df["label"].to_numpy()

    print("\nClassification Report:")
    print(classification_report(
        actual,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    cm = confusion_matrix(actual, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.show()

    fpr, tpr, _ = roc_curve(actual, probabilities)
    auc_score = roc_auc_score(actual, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.show()

    print(f"ROC-AUC: {auc_score:.4f}")


if __name__ == "__main__":
    main()