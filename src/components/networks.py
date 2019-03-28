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
        self.discriminator_spatial_a = Discriminator('discriminator_a', is_train=placeholders.is_train,
                                                     norm='instance', activation='leaky')
        self.discriminator_spatial_b = Discriminator('discriminator_b', is_train=placeholders.is_train,
                                                     norm='instance', activation='leaky')

        self.discriminator_temporal = Discriminator('discriminator_temp', is_train=placeholders.is_train,
                                                    norm='instance', activation='leaky')


    def init_fnet(self, placeholders: Placeholders):

        flows = self.get_flows_to_middle_frame(placeholders.image_warp_input)
        self.warped_real = self.warp_to_middle_frame(placeholders.image_warp_input, flows)
        self.warped_fake = self.warp_to_middle_frame(placeholders.fake_warp_input, flows)


    def get_flows_to_middle_frame(self, frame_sequence):
        frame_sequence = (frame_sequence+1)/2
        previous = frame_sequence[:, 0]
        current = frame_sequence[:, 1]
        next = frame_sequence[:, 2]

        backwards_flow = self.get_flow(previous, current)
        forwards_flow = self.get_flow(next, current)

        return  backwards_flow, forwards_flow

    def get_flow(self, first, second):
        input = tf.concat([first, second], axis=-1)
        with tf.variable_scope('fnet',reuse=tf.AUTO_REUSE):
            flow = fnet(input)
        return tf.image.resize_images(flow,first.shape.as_list()[1:-1])

    def warp_to_middle_frame(self, frame_sequence, flows):
        previous = frame_sequence[:, 0]
        current = frame_sequence[:, 1]
        next = frame_sequence[:, 2]

        backwards_flow, forwards_flow = flows

        previous_warped = tf.contrib.image.dense_image_warp(previous, backwards_flow)
        next_warped = tf.contrib.image.dense_image_warp(next, forwards_flow)

        return tf.stack([previous_warped,current,next_warped],axis=1)

    def warp_frame(self, frame, flow):
        return tf.contrib.image.dense_image_warp(frame, flow)

    def apply_inference_on_multiframe(self, frames, generator):

        '''
        TODO
        1. generate first fake frame with black frame as input (tf.concat([first, tf.constant(-1.0, shape=[b,256, 235,3])], -1) )
        2. compute flow from first to second
        3. warp first output frame to space of second
        4. generate second fake with tf.concat([second, warped_first_fake))

        5. 2-4. analogue for third frame
        '''
        first = frames[:, 0]
        second = frames[:, 1]
        third = frames[:, 2]

        first_input = tf.concat([first, tf.constant(-1.0, shape=first.get_shape().as_list())], axis=-1)
        first_flow = self.get_flow(first, second)
        first_output = generator(first_input)
        first_warped = self.warp_frame(first_output, first_flow)

        second_input = tf.concat([second, first_warped], axis=-1)
        second_flow = self.get_flow(second, third)
        second_output = generator(second_input)
        second_warped = self.warp_frame(second_output, second_flow)

        third_input = tf.concat([third, second_warped], axis=-1)
        third_output = generator(third_input)

        return tf.stack([first_output,second_output,third_output], axis=1)


        #frame_tensor_shape = frames.get_shape().as_list()
        #target_shape = frame_tensor_shape.copy()
        #target_shape[0] *= target_shape[1]
        #target_shape.pop(1)

        #return tf.reshape(generator(tf.reshape(frames, target_shape)),frame_tensor_shape)

