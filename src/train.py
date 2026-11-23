from pathlib import Path
import tensorflow as tf

from data_utils import load_dataframes
from preprocessing import (
    split_data,
    create_tf_dataset,
    calculate_class_weights
)
from model import build_custom_cnn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_custom_cnn.keras"

EPOCHS = 20


def main():
    print("Loading dataset...")

    development_df, test_df, dataset_root = load_dataframes(
        PROJECT_ROOT
    )

    print("Dataset:", dataset_root)
    print("Development images:", len(development_df))
    print("Test images:", len(test_df))

    train_df, val_df = split_data(development_df)

    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))

    train_df.to_csv(OUTPUT_DIR / "train_manifest.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation_manifest.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_manifest.csv", index=False)

    train_dataset = create_tf_dataset(train_df, training=True)
    val_dataset = create_tf_dataset(val_df)
    test_dataset = create_tf_dataset(test_df)

    class_weights = calculate_class_weights(train_df)
    print("Class weights:", class_weights)

    model = build_custom_cnn()

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=0.000001,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / "training_log.csv")
        )
    ]

    print("Starting CNN training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(MODEL_DIR / "final_custom_cnn.keras")

    print("Best model saved to:", MODEL_PATH)

    print("Evaluating on untouched test set...")
    best_model = tf.keras.models.load_model(MODEL_PATH)
    results = best_model.evaluate(test_dataset, return_dict=True)

    print("Test results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    pd_history = __import__("pandas").DataFrame(history.history)
    pd_history.to_csv(OUTPUT_DIR / "training_history.csv", index=False)


if __name__ == "__main__":
    main()