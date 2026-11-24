
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

from load_dataset import class_names
from preprocessing import IMAGE_SIZE


print("Wczytywanie modelu...")
model = tf.keras.models.load_model("models/best_model.keras")
print("Model gotowy!")


root = tk.Tk()
root.title("Rozpoznawanie polskich znaków drogowych")
root.geometry("750x650")
root.configure(bg="#f0f2f5")
root.resizable(False, False)


title = tk.Label(
    root,
    text="ROZPOZNAWANIE ZNAKÓW DROGOWYCH",
    font=("Arial", 19, "bold"),
    bg="#f0f2f5",
    fg="#1f2937"
)
title.pack(pady=25)


subtitle = tk.Label(
    root,
    text="System klasyfikacji znaków drogowych",
    font=("Arial", 11),
    bg="#f0f2f5",
    fg="#6b7280"
)
subtitle.pack(pady=5)


image_label = tk.Label(
    root,
    text="Wybierz zdjęcie znaku drogowego",
    font=("Arial", 13),
    bg="white",
    fg="#9ca3af",
    width=45,
    height=12,
    relief="solid",
    borderwidth=1
)
image_label.pack(pady=20)



result_label = tk.Label(
    root,
    text="Oczekiwanie na zdjęcie...",
    font=("Arial", 16, "bold"),
    bg="#f0f2f5",
    fg="#374151"
)
result_label.pack(pady=10)


confidence_label = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    bg="#f0f2f5",
    fg="#2563eb"
)
confidence_label.pack(pady=5)


def upload_image():
    global photo

    file_path = filedialog.askopenfilename(
        title="Wybierz zdjęcie znaku drogowego",
        filetypes=[
            ("Zdjęcia", "*.jpg *.jpeg *.png *.bmp"),
            ("Wszystkie pliki", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        original_image = Image.open(file_path).convert("RGB")

        preview = original_image.copy()
        preview.thumbnail((400, 300))

        photo = ImageTk.PhotoImage(preview)

        image_label.config(
            image=photo,
            text="",
            width=400,
            height=300
        )

        
        image = tf.io.read_file(file_path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize_with_pad(
            image,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1]
        )

        image = tf.cast(image, tf.float32) / 255.0

        image = tf.expand_dims(image, axis=0)

        
        predictions = model.predict(image, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]

        confidence = predictions[predicted_index] * 100

        
        result_label.config(
            text=f"Rozpoznany znak: {predicted_class}",
            fg="#15803d"
        )

        confidence_label.config(
            text=f"Pewność rozpoznania: {confidence:.2f}%"
        )

    except Exception as error:
        messagebox.showerror(
            "Błąd",
            f"Nie udało się rozpoznać zdjęcia:\n{error}"
        )



upload_button = tk.Button(
    root,
    text="Wgraj zdjęcie",
    font=("Arial", 13, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=25,
    pady=12,
    command=upload_image
)
upload_button.pack(pady=20)


root.mainloop()
