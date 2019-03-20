import os.path
import tensorflow as tf
from src.components.savers import Saver
from src.nets.generator import Generator


class InferenceMachine:
    def __init__(self, height, width, model_dir):

        self.create_graph(model_dir, height, width)
        self.create_session()
        self.restore_generator_weights()

    def create_graph(self, model_dir, height, width):

        self.create_placeholders(height, width)
        self.create_generators()
        self.define_generator_output()
        self.create_savers(model_dir)

    def create_placeholders(self, height, width):
        self.image_a = tf.placeholder(tf.float32, [1, height, width, 3], name='image_a')
        self.image_b = tf.placeholder(tf.float32, [1, height, width, 3], name='image_b')

    def create_generators(self):
        self.generator_ab = Generator('generator_ab', norm='instance',
                                      activation='relu', is_train=None)
        self.generator_ba = Generator('generator_ba', norm='instance',
                                      activation='relu', is_train=None)

    def define_generator_output(self):
        self.image_ab = self.generator_ab(self.image_a)
        self.image_ba = self.generator_ba(self.image_b)

    def create_savers(self, model_dir):
        self.generator_ab_saver = Saver(self.generator_ab.var_list,
                                        save_path=os.path.join(model_dir, self.generator_ab.name), name="Generator AB")
        self.generator_ba_saver = Saver(self.generator_ba.var_list,
                                        save_path=os.path.join(model_dir, self.generator_ba.name), name="Generator BA")

    def create_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def restore_generator_weights(self):
        self.generator_ab_saver.load(self.sess)
        self.generator_ba_saver.load(self.sess)

    def __del__(self):
        self.sess.close()

    def single_image_inference(self, input_image, forwards):

        graph = self._get_inference_graph(forwards)

        result = self.sess.run(graph, feed_dict={self.image_a: [input_image],
                                                 self.image_b: [input_image]})[0]
        return result

    def multi_frame_inference(self, input_frames, forwards):
        result = []

        graph = self._get_inference_graph(forwards)
        for frame in input_frames:
            result.append(self.sess.run(graph, feed_dict={self.image_a: [frame],
                                                          self.image_b: [frame]})[0])
        return result

    def _get_inference_graph(self, forwards):
        if forwards:
            graph = self.image_ab
        else:
            graph = self.image_ba
        return graph
