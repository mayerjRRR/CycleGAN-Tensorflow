import logging
import os


# start logging
import tensorflow as tf

logging.info("Start CycleGAN")
logger = logging.getLogger('cycle-gan')
logger.setLevel(logging.INFO)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_image(image_size, filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_normalized = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image_normalized = (image_normalized * 2) - 1
    image_resized = tf.image.resize_images(image_normalized, [image_size, image_size])
    return image_resized