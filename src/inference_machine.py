import tensorflow as tf

from src.cycle_gan import CycleGan


class InferenceMachine():
    def __init__(self, height, width, model_dir):

        self.model = CycleGan(image_height=height, image_width=width, batch_size=1)

        weight_initiallizer = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init_op)
        self.sess.run(local_init_op)
        weight_initiallizer.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def __del__(self):
        self.sess.close()


    def single_image_inference(self, input_image, forwards):

        graph = self._get_inference_graph(forwards)

        result = self.sess.run(graph, feed_dict={self.model.placeholders.image_a: [input_image],
                                                self.model.placeholders.image_b: [input_image],
                                                                   self.model.placeholders.is_train: False})[0]
        return result

    def multi_frame_inference(self, input_frames, forwards):
        result = []

        graph = self._get_inference_graph(forwards)
        for frame in input_frames:
            result.append(self.sess.run(graph, feed_dict={self.model.placeholders.image_a: [frame],
                                                         self.model.placeholders.image_b: [frame],
                                                                         self.model.placeholders.is_train: False})[0])
        return result

    def _get_inference_graph(self, forwards):
        if forwards:
            graph = self.model.images.image_ab
        else:
            graph = self.model.images.image_ba
        return graph
