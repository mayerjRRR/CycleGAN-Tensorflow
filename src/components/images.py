import tensorflow as tf

from src.components.networks import Networks
from src.components.placeholders import Placeholders
from src.utils.tensor_ops import extract_frames_from_channels, layer_frames_in_channels


class Images:

    def __init__(self, placeholders: Placeholders, networks: Networks, image_shape, batch_size, augment_shape):
        self.augment_data_for_training(placeholders, image_shape, batch_size, augment_shape)
        self.generate_fake_images(networks)
        self.generate_fake_frames(networks)

    # TODO: Move to Dataset maybe
    def augment_data_for_training(self, placeholders: Placeholders, image_shape, batch_size, augment_shape):
        def augment_image(image):
            image = tf.image.resize_images(image, augment_shape)
            #TODO: Do this nicer
            image = tf.random_crop(image, [batch_size] + image_shape[0:-1]+[image.get_shape().as_list()[-1]])
            image = tf.map_fn(tf.image.random_flip_left_right, image)
            return image

        self.image_a = tf.cond(placeholders.is_train,
                               lambda: augment_image(placeholders.image_a),
                               lambda: placeholders.image_a)
        self.image_b = tf.cond(placeholders.is_train,
                               lambda: augment_image(placeholders.image_b),
                               lambda: placeholders.image_b)

        self.frames_a = tf.cond(placeholders.is_train,
                                lambda: extract_frames_from_channels(
                                    augment_image(layer_frames_in_channels(placeholders.frames_a))),
                                lambda: placeholders.frames_a)
        self.frames_b = tf.cond(placeholders.is_train,
                                lambda: extract_frames_from_channels(
                                    augment_image(layer_frames_in_channels(placeholders.frames_b))),
                                lambda: placeholders.frames_b)

    def generate_fake_images(self, networks: Networks):
        self.image_ab = networks.generator_ab(self.image_a)
        self.image_ba = networks.generator_ba(self.image_b)
        self.image_bab = networks.generator_ab(self.image_ba)
        self.image_aba = networks.generator_ba(self.image_ab)

    def generate_fake_frames(self, networks: Networks):
        self.flows_a = networks.get_flows(self.frames_a)
        self.flows_b = networks.get_flows(self.frames_b)

        self.warped_frames_a = networks.warp(self.frames_a, self.flows_a)
        self.warped_frames_b = networks.warp(self.frames_b, self.flows_b)

        self.frames_ab = networks.apply_inference_on_multiframe(self.frames_a, networks.generator_ab)
        self.frames_ba = networks.apply_inference_on_multiframe(self.frames_b, networks.generator_ba)

        self.warped_frames_ab = networks.warp(self.frames_ab, self.flows_a)
        self.warped_frames_ba = networks.warp(self.frames_ba, self.flows_b)

        self.frames_aba = networks.apply_inference_on_multiframe(self.frames_ab, networks.generator_ba)
        self.frames_bab = networks.apply_inference_on_multiframe(self.frames_ba, networks.generator_ab)
