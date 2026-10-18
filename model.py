import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():

    model = models.Sequential([

        layers.Input(
            shape=(1024, 1)
        ),

        layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Conv1D(
            filters=64,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling1D(
            pool_size=2
        ),

        layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.GlobalAveragePooling1D(), 

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.3),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model