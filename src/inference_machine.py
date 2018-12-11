import tensorflow as tf

from src.cycle_gan import CycleGan
from src.utils.fast_saver import FastSaver


class InferenceMachine():
    def __init__(self, height, width, model_dir):
        self.model = CycleGan(image_height=height, image_width=width, batch_size=1)
        variables_to_save = tf.global_variables()
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
        saver = FastSaver(variables_to_save)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     tf.get_variable_scope().name)

        logdir = model_dir
        summary_writer = tf.summary.FileWriter(logdir)

        def init_fn(sess):
            sess.run(init_all_op)
        #sm = tf.train.SessionManager()
        self.sv = tf.train.Supervisor(is_chief=True,
                                 logdir=logdir,
                                 saver=saver,
                                 summary_op=None,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 summary_writer=summary_writer,
                                 ready_op=tf.report_uninitialized_variables(variables_to_save),
                                 global_step=self.model.placeholders.global_step,
                                 save_model_secs=300,
                                 save_summaries_secs=30)


    def single_image_inference(self, input_image, forwards):

        graph = self._get_inference_graph(forwards)

        with self.sv.managed_session() as sess:
            result = sess.run(graph, feed_dict={self.model.placeholders.image_a: [input_image],
                                                                   self.model.placeholders.is_train: False})[0]
        return result

    def backward_inference(self, input_image):
        with self.sv.managed_session() as sess:
            result = sess.run(self.model.images.image_ba, feed_dict={self.model.placeholders.image_b: [input_image],
                                                                   self.model.placeholders.is_train: False})[0]
        return result

    def multi_frame_inference(self, input_frames, forwards):
        result = []

        graph = self._get_inference_graph(forwards)

        with self.sv.managed_session() as sess:
            for frame in input_frames:
                result.append(sess.run(graph, feed_dict={self.model.placeholders.image_a: [frame],
                                                                         self.model.placeholders.is_train: False})[0])
        return result

    def _get_inference_graph(self, forwards):
        if forwards:
            graph = self.model.images.image_ab
        else:
            graph = self.model.images.image_ba
        return graph
