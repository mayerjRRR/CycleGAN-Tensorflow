from src.components.placeholders import Placeholders
from src.nets.discriminator import Discriminator
from src.nets.generator import Generator
from fnet.fnet import fnet
import tensorflow as tf


class Networks:
    def __init__(self, placeholders: Placeholders):
        self.init_generators(placeholders)
        self.init_discriminators(placeholders)
        self.init_fnet(placeholders)

    def init_generators(self, placeholders: Placeholders):
        self.generator_ab = Generator('generator_ab', is_train=placeholders.is_train, norm='instance',
                                      activation='relu')
        self.generator_ba = Generator('generator_ba', is_train=placeholders.is_train, norm='instance',
                                      activation='relu')

    def init_discriminators(self, placeholders: Placeholders):
        self.discriminator_a = Discriminator('discriminator_a', is_train=placeholders.is_train,
                                             norm='instance', activation='leaky')
        self.discriminator_b = Discriminator('discriminator_b', is_train=placeholders.is_train,
                                             norm='instance', activation='leaky')

    def init_fnet(self, placeholders: Placeholders):
        with tf.variable_scope('fnet'):

            preprocessed_input = (placeholders.fnet_placeholder+1)/2
            self.fnet = fnet(preprocessed_input)
        with tf.variable_scope('warp_image'):
            est_flow = tf.image.resize_images(self.fnet, placeholders.fnet_placeholder.shape.as_list()[1:-1])
            self.pre_input_warp = tf.contrib.image.dense_image_warp(placeholders.fnet_placeholder[:, :, :, :3],
                                                                    est_flow)