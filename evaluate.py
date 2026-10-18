import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# Load test data
# -----------------------------

data = np.load("data/test.npz")

X_test = data["X"]
y_test = data["y"]


# CNN input
X_test = X_test.reshape(-1, 1024, 1)


# -----------------------------
# Load trained model
# -----------------------------

model = load_model("chacha_dl_model.keras")


# -----------------------------
# Predictions
# -----------------------------

probabilities = model.predict(X_test,verbose=0).flatten()   #model produces possibilities for each sample, flatten() converts the 2D array to 1D array


predictions = (probabilities >= 0.5).astype(int)   # Convert probabilities to binary predictions


# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(y_test,predictions)

precision = precision_score(y_test,predictions)

recall = recall_score(y_test,predictions)

f1 = f1_score(y_test,predictions)

cm = confusion_matrix(y_test,predictions)


print("\n==============================")
print("ChaCha Reduced-Round Results")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Random","Differential"]
    )
)