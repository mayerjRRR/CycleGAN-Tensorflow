import tensorflow as tf

from src.components.networks import Networks
from src.components.placeholders import Placeholders
from src.utils.tensor_ops import extract_frames_from_channels, layer_frames_in_channels
from src.utils.warp_utils import get_flows_to_middle_frame, warp_to_middle_frame, apply_inference_on_multiframe, \
    pingpongify, unpingpongify, get_fake_generator_input


class Images:

    def __init__(self, placeholders: Placeholders, networks: Networks, augment_shape, image_shape):
        self.define_input(placeholders, augment_shape, image_shape)
        self.define_fake_images(networks)
        self.define_fake_frames(networks)

    def define_input(self, placeholders: Placeholders, augment_shape, image_shape):
        def augment_image(image):
            downscaled_image = tf.image.resize_image_with_crop_or_pad(image, augment_shape[0],augment_shape[1])
            cropped_image = tf.random_crop(downscaled_image, [image.get_shape().as_list()[0]]+image_shape[0:2]+[image.get_shape().as_list()[-1]])
            flipped_image = tf.map_fn(tf.image.random_flip_left_right, cropped_image)
            return flipped_image
        self.image_a = tf.cond(placeholders.is_train,
                               lambda: augment_image(placeholders.image_a),
                               lambda: tf.image.resize_images(placeholders.image_a, image_shape[0:2]))
        self.image_b = tf.cond(placeholders.is_train,
                               lambda: augment_image(placeholders.image_b),
                               lambda: tf.image.resize_images(placeholders.image_b, image_shape[0:2]))

        self.frames_a = tf.cond(placeholders.is_train,
                                lambda: extract_frames_from_channels(
                                    augment_image(layer_frames_in_channels(placeholders.frames_a))),
                                lambda: extract_frames_from_channels(tf.image.resize_images(layer_frames_in_channels(placeholders.frames_a),image_shape[0:2])))
        self.frames_b = tf.cond(placeholders.is_train,
                                lambda: extract_frames_from_channels(
                                    augment_image(layer_frames_in_channels(placeholders.frames_b))),
                                lambda: extract_frames_from_channels(tf.image.resize_images(layer_frames_in_channels(placeholders.frames_b),image_shape[0:2])))

    def define_fake_images(self, networks: Networks):
        self.image_ab = networks.generator_ab(get_fake_generator_input(self.image_a))
        self.image_ba = networks.generator_ba(get_fake_generator_input(self.image_b))

        self.image_bab  = networks.generator_ab(get_fake_generator_input(self.image_ba))
        self.image_aba  = networks.generator_ba(get_fake_generator_input(self.image_ab))

    def define_fake_frames(self, networks: Networks):

        self.flows_a = get_flows_to_middle_frame(self.frames_a)
        self.flows_b = get_flows_to_middle_frame(self.frames_b)

        self.warped_frames_a = warp_to_middle_frame(self.frames_a, self.flows_a)
        self.warped_frames_b = warp_to_middle_frame(self.frames_b, self.flows_b)

        self.pingpong_frames_ab = apply_inference_on_multiframe(pingpongify(self.frames_a), networks.generator_ab)
        self.pingpong_frames_ba = apply_inference_on_multiframe(pingpongify(self.frames_b), networks.generator_ba)

        self.frames_ab = unpingpongify(self.pingpong_frames_ab)[0]
        self.frames_ba = unpingpongify(self.pingpong_frames_ba)[0]

        self.code_frames_ab = networks.generator_ab(get_fake_generator_input(self.frames_a[:,0]), return_code_layer=True)
        self.code_frames_ba = networks.generator_ba(get_fake_generator_input(self.frames_b[:,0]), return_code_layer=True)

        self.warped_frames_ab = warp_to_middle_frame(self.frames_ab, self.flows_a)
        self.warped_frames_ba = warp_to_middle_frame(self.frames_ba, self.flows_b)

        self.frames_aba = apply_inference_on_multiframe(self.frames_ab, networks.generator_ba)
        self.frames_bab = apply_inference_on_multiframe(self.frames_ba, networks.generator_ab)

        self.code_frames_aba = networks.generator_ba(get_fake_generator_input(self.frames_ab[:,0]), return_code_layer=True)
        self.code_frames_bab = networks.generator_ab(get_fake_generator_input(self.frames_ba[:,0]), return_code_layer=True)
