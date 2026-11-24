
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import test_dataset, class_names

# Wczytanie najlepszego modelu
print("Wczytywanie modelu...")
model = tf.keras.models.load_model("models/best_model.keras")

# Ocena modelu na zbiorze testowym
print("\nOcena modelu na zbiorze testowym...")
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nDokładność testowa: {test_accuracy * 100:.2f}%")
print(f"Strata testowa: {test_loss:.4f}")

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print("\nRaport klasyfikacji:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Przewidziana klasa")
plt.ylabel("Prawdziwa klasa")
plt.title("Macierz pomyłek")

plt.tight_layout()

plt.xlabel("Przewidziana klasa")
plt.ylabel("Prawdziwa klasa")
plt.title("Macierz pomyłek")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")
plt.close()

print("\nMacierz pomyłek zapisana!")
print("OCENA MODELU ZAKOŃCZONA!")
