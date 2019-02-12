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
       # with tf.variable_scope('fnet'):
       #     preprocessed_input = (placeholders.fnet_input_placeholder + 1) / 2
       #     self.fnet = fnet(preprocessed_input)
       # with tf.variable_scope('warp_image'):
       #     est_flow = tf.image.resize_images(self.fnet, placeholders.fnet_input_placeholder.shape.as_list()[1:-1])
       #     self.pre_input_warp = tf.contrib.image.dense_image_warp(placeholders.fnet_input_placeholder[:, :, :, :3],
       #           est_flow)
        flows = self.get_flows(placeholders.image_warp_input)
        self.warped_real = self.warp(placeholders.image_warp_input, flows)
        self.warped_fake = self.warp(placeholders.fake_warp_input, flows)


    def get_flows(self, frame_sequence):
        frame_sequence = (frame_sequence+1)/2
        previous = frame_sequence[:, 0]
        current = frame_sequence[:, 1]
        next = frame_sequence[:, 2]

        print(current.get_shape())

        backwards_flow = self.get_flow(previous, current)
        forwards_flow = self.get_flow(next, current)

        return  backwards_flow, forwards_flow

    def get_flow(self, first, second):
        input = tf.concat([first, second], axis=-1)

        print("Shape: "+str(input.get_shape()))

        with tf.variable_scope('fnet',reuse=tf.AUTO_REUSE):
            flow = fnet(input)
        return tf.image.resize_images(flow,first.shape.as_list()[1:-1])

    def warp(self, frame_sequence, flows):
        previous = frame_sequence[:, 0]
        current = frame_sequence[:, 1]
        next = frame_sequence[:, 2]

        backwards_flow, forwards_flow = flows

        previous_warped = tf.contrib.image.dense_image_warp(previous, backwards_flow)
        next_warped = tf.contrib.image.dense_image_warp(next, forwards_flow)

        return tf.stack([previous_warped,current,next_warped],axis=1)
