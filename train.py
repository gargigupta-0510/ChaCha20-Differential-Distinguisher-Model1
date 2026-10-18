import numpy as np
from sklearn.model_selection import train_test_split
from model import build_model


# Load dataset
data = np.load("data/train.npz")

X = data["X"]
y = data["y"]


# Create validation set
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y              #keeps the class proportion approximately equal in both train and validation sets
)


# CNN expects:
# samples × 1024 × 1

X_train = X_train.reshape(-1, 1024, 1)    #(samples, length, channels) for CNN input, -1 means numpy will calculate the number of samples automatically based on the other dimensions. In this case, it will be 80% of the original dataset size.

X_val = X_val.reshape(-1, 1024, 1)


# Build model
model = build_model()

model.summary()


# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val,y_val),
    epochs=15,
    batch_size=256,
    verbose=1
)


# Save
model.save(
    "chacha_dl_model.keras"
)

print("\nModel saved.")