
from pathlib import Path
import tensorflow as tf
from preprocessing import preprocess_image

DATASET_DIR = Path("prepared_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BATCH_SIZE = 32

class_names = sorted(
    folder.name
    for folder in (DATASET_DIR / "train").iterdir()
    if folder.is_dir()
)

class_to_index = {
    name: index for index, name in enumerate(class_names)
}


def load_split(split_name):
    split_dir = DATASET_DIR / split_name

    image_paths = []
    labels = []

    for class_name in class_names:
        class_dir = split_dir / class_name

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(class_to_index[class_name])

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if split_name == "train":
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=42
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = load_split("train")
validation_dataset = load_split("validation")
test_dataset = load_split("test")

print("Zbiory danych zostały wczytane!")
print("Liczba klas:", len(class_names))
print("Nazwy klas:", class_names)

for images, labels in train_dataset.take(1):
    print("Rozmiar partii zdjęć:", images.shape)
    print("Rozmiar partii etykiet:", labels.shape)
    print("Minimalna wartość piksela:", tf.reduce_min(images).numpy())
    print("Maksymalna wartość piksela:", tf.reduce_max(images).numpy())
