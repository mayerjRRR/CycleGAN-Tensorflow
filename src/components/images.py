import tensorflow as tf

from src.components.networks import Networks
from src.components.placeholders import Placeholders


class Images:

    def __init__(self, placeholders: Placeholders, networks: Networks, image_shape, batch_size, augment_shape):
        self.augment_data_for_training(placeholders, image_shape, batch_size, augment_shape)
        self.generate_fake_images(networks)

    #TODO: Move to Dataset maybe
    def augment_data_for_training(self, placeholders: Placeholders, image_shape, batch_size, augment_shape):
        def augment_image(image):
            image = tf.image.resize_images(image, augment_shape)
            image = tf.random_crop(image, [batch_size] + image_shape)
            #TODO: Turn off for sequence
            image = tf.map_fn(tf.image.random_flip_left_right, image)
            return image

        self.image_a = tf.cond(placeholders.is_train,
                               lambda: augment_image(placeholders.image_a),
                               lambda: placeholders.image_a)
        self.image_b = tf.cond(placeholders.is_train,
                               lambda: augment_image(placeholders.image_b),
                               lambda: placeholders.image_b)

    def generate_fake_images(self, networks: Networks):
        self.image_ab = networks.generator_ab(self.image_a)
        self.image_ba = networks.generator_ba(self.image_b)
        self.image_bab = networks.generator_ab(self.image_ba)
        self.image_aba = networks.generator_ba(self.image_ab)