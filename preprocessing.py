
import tensorflow as tf

IMAGE_SIZE = (64, 64)

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    #Zachowanie proporcji i uzupełnienie pustych przestrzeni
    image = tf.image.resize_with_pad(
        image,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    )

    #Normalizacja pikseli do zakresu 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label
