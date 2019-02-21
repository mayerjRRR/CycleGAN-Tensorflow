import tensorflow as tf

from src.components.networks import Networks
from src.components.placeholders import Placeholders


class Images:

    def __init__(self, placeholders: Placeholders, networks: Networks, image_shape, batch_size, augment_shape):
        self.augment_data_for_training(placeholders, image_shape, batch_size, augment_shape)
        self.generate_fake_images(networks)
        self.generate_fake_frames(networks, placeholders)

    #TODO: Move to Dataset maybe
    def augment_data_for_training(self, placeholders: Placeholders, image_shape, batch_size, augment_shape):
        def augment_image(image):
            #image = tf.image.resize_images(image, augment_shape)
            #image = tf.random_crop(image, [batch_size] + image_shape)
            #TODO: Turn off for sequence, find a way to reactivate
            #image = tf.map_fn(tf.image.random_flip_left_right, image)
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

    def generate_fake_frames(self, networks: Networks, placeholders:Placeholders):
        self.frames_a = placeholders.frames_a
        self.frames_b = placeholders.frames_b

        self.flows_a = networks.get_flows(placeholders.frames_a)
        self.flows_b = networks.get_flows(placeholders.frames_b)

        self.warped_frames_a = networks.warp(placeholders.frames_a, self.flows_a)
        self.warped_frames_b = networks.warp(placeholders.frames_b, self.flows_b)

        self.frames_ab = networks.apply_inference_on_multiframe(placeholders.frames_a, networks.generator_ab)
        self.frames_ba = networks.apply_inference_on_multiframe(placeholders.frames_b, networks.generator_ba)

        self.warped_frames_ab = networks.warp(self.frames_ab, self.flows_a)
        self.warped_frames_ba = networks.warp(self.frames_ba, self.flows_b)

        self.frames_aba = networks.apply_inference_on_multiframe(self.frames_ab, networks.generator_ba)
        self.frames_bab = networks.apply_inference_on_multiframe(self.frames_ba, networks.generator_ab)
