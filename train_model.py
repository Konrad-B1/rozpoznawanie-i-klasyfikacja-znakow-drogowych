
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from load_dataset import (
    train_dataset,
    validation_dataset,
    class_names
)

NUM_CLASSES = len(class_names)
EPOCHS = 30

Path("models").mkdir(exist_ok=True)

#Budowa sieci CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# Kompilacja modelu
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Wyświetlenie architektury
model.summary()

#Zapisywanie najlepszego modelu
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

#Trenowanie
print("\nROZPOCZYNAM TRENING SIECI CNN\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nZapisywanie modelu...")

model.save("models/final_model.keras")

print("Model zapisany!")

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Trening")
plt.plot(history.history["val_accuracy"], label="Walidacja")
plt.title("Dokładność modelu")
plt.xlabel("Epoka")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("models/accuracy.png")
plt.close()

print("Wykres dokładności zapisany!")

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Trening")
plt.plot(history.history["val_loss"], label="Walidacja")
plt.title("Strata modelu")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("models/loss.png")
plt.close()

print("Wykres straty zapisany!")

print("\nTRENING ZAKOŃCZONY!")