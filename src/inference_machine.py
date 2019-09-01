import os.path
import tensorflow as tf
from src.components.savers import Saver
from src.nets.generator import Generator
import numpy as np

from src.utils.warp_utils import recurrent_inference


class InferenceMachine:
    def __init__(self, height, width, model_dir, unet, no_temp):

        self.create_graph(model_dir, height, width, unet)
        self.create_session()
        self.restore_generator_weights()

        self.no_temp = no_temp

        self.last_frame_array = None
        self.last_result_array = None

    def create_graph(self, model_dir, height, width, unet):

        self.create_placeholders(height, width)
        self.create_generators(unet)
        self.define_generator_output()
        self.create_savers(model_dir)

    def create_placeholders(self, height, width):
        self.current_frame = tf.placeholder(tf.float32, [1, height, width, 3], name='current')
        self.last_frame = tf.placeholder(tf.float32, [1, height, width, 3], name='last')
        self.last_result = tf.placeholder(tf.float32, [1, height, width, 3], name='last_result')

    def create_generators(self, unet):
        self.generator_ab = Generator('generator_ab', norm='instance',
                                      activation='relu', is_train=None, unet=unet)
        self.generator_ba = Generator('generator_ba', norm='instance',
                                      activation='relu', is_train=None, unet=unet)

    def define_generator_output(self):
        self.image_ab = recurrent_inference(self.generator_ab, self.current_frame, self.last_frame, self.last_result)
        self.image_ba = recurrent_inference(self.generator_ba, self.current_frame, self.last_frame, self.last_result)

    def create_savers(self, model_dir):
        self.generator_ab_saver = Saver(self.generator_ab.var_list,
                                        save_path=os.path.join(model_dir, self.generator_ab.name), name="Generator AB")
        self.generator_ba_saver = Saver(self.generator_ba.var_list,
                                        save_path=os.path.join(model_dir, self.generator_ba.name), name="Generator BA")

        fnet_variable_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
        self.fnet_saver = Saver(fnet_variable_list, save_path='./fnet',
                                name="FNet")

    def create_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def restore_generator_weights(self):
        self.generator_ab_saver.load(self.sess)
        self.generator_ba_saver.load(self.sess)
        self.fnet_saver.load(self.sess)

    def __del__(self):
        self.sess.close()

    def recurrent_inference(self, input_image, forwards):

        if self.last_frame_array is None:
            self.last_frame_array = np.full_like(input_image, -1.0)
            self.last_result_array = np.full_like(input_image, -1.0)

        graph = self._get_inference_graph(forwards)

        if not self.no_temp:
            result = self.sess.run(graph, feed_dict={self.current_frame: [input_image],
                                                     self.last_frame: [self.last_frame_array],
                                                     self.last_result: [self.last_result_array]})[0]
        else:
            result = self.sess.run(graph, feed_dict={self.current_frame: [input_image],
                                                     self.last_frame: [input_image],
                                                     self.last_result: [input_image]})[0]

        self.last_frame_array = input_image
        self.last_result_array = result
        return result

    def _get_inference_graph(self, forwards):
        if forwards:
            graph = self.image_ab
        else:
            graph = self.image_ba
        return graph
