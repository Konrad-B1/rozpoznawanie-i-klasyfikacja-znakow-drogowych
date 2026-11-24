
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import shutil

SOURCE_DIR = Path("classification")
OUTPUT_DIR = Path("prepared_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not SOURCE_DIR.exists():
        print("Nie znaleziono folderu classification!")
        return

    if OUTPUT_DIR.exists():
        print("")
        print("")
        return

    files = []
    labels = []

    for class_dir in sorted(SOURCE_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image in images:
            files.append(image)
            labels.append(class_dir.name)

    counts = Counter(labels)

    if min(counts.values()) < 3:
        print("Za mało zdjęć w jednej z klas!")
        return

    train_files, temp_files, train_labels, temp_labels = (
        train_test_split(
            files,
            labels,
            test_size=0.30,
            random_state=42,
            stratify=labels
        )
    )

    val_files, test_files, val_labels, test_labels = (
        train_test_split(
            temp_files,
            temp_labels,
            test_size=0.50,
            random_state=42,
            stratify=temp_labels
        )
    )

    datasets = {
        "train": (train_files, train_labels),
        "validation": (val_files, val_labels),
        "test": (test_files, test_labels)
    }

    for split, (images, image_labels) in datasets.items():
        for image, label in zip(images, image_labels):
            destination = OUTPUT_DIR / split / label
            destination.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                image,
                destination / image.name
            )

    print("\nPODZIAL ZBIORU ZAKONCZONY")
    print("=" * 45)

    for split, (images, _) in datasets.items():
        print(f"{split:12}: {len(images)} zdjęć")

    print("=" * 45)
    print("Liczba klas:", len(counts))
    print("Łączna liczba zdjęć:", len(files))


if __name__ == "__main__":
    main()
