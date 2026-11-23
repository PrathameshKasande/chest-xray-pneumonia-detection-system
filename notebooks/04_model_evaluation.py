
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load test dataset
X_test = np.load("processed_data/X_test.npy")
y_test = np.load("processed_data/y_test.npy")

# Load your trained CNN model
model = tf.keras.models.load_model(
    "models/best_custom_cnn.keras"
)

# Generate predictions
probabilities = model.predict(X_test)
predictions = (probabilities.reshape(-1) >= 0.5).astype(int)

# Calculate metrics
print("\nMODEL EVALUATION")
print("----------------------------")

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, predictions, zero_division=0))

# Classification report
print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

display.plot(cmap="Blues")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.show()
