import tensorflow as tf


class Placeholders:
    def __init__(self, batch_size, training_shape, input_shape, frame_sequence_length):
        self.frame_sequence_length = frame_sequence_length

        self.init_training_placeholders()
        self.init_real_placeholders(batch_size, input_shape)
        self.init_fake_history_placeholders(batch_size, training_shape)
        self.init_tb_placeholders()

    def init_training_placeholders(self):
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.global_step = tf.train.get_or_create_global_step(
            graph=None)

    def init_real_placeholders(self, batch_size, input_shape):
        self.image_a = tf.placeholder(tf.float32, [batch_size] + input_shape, name='image_a')
        self.image_b = tf.placeholder(tf.float32, [batch_size] + input_shape, name='image_b')

        self.frames_a = tf.placeholder(tf.float32, [batch_size,self.frame_sequence_length ] + input_shape, name='frames_a')
        self.frames_b = tf.placeholder(tf.float32, [batch_size,self.frame_sequence_length ] + input_shape, name='frames_b')

    def init_fake_history_placeholders(self, batch_size, image_shape):
        self.history_fake_a = tf.placeholder(tf.float32, [batch_size] + image_shape, name='history_fake_a')
        self.history_fake_b = tf.placeholder(tf.float32, [batch_size] + image_shape, name='history_fake_b')

        self.history_fake_temp_frames_a = tf.placeholder(tf.float32, [batch_size, 3 ] + image_shape, name='history_fake_a')
        self.history_fake_temp_frames_b = tf.placeholder(tf.float32, [batch_size, 3 ] + image_shape, name='history_fake_b')

    def init_tb_placeholders(self):
        self.tb_spatial_a = tf.placeholder(tf.float32, name='spatial_tb_a')
        self.tb_spatial_b = tf.placeholder(tf.float32, name='spatial_tb_b')
        self.tb_temporal = tf.placeholder(tf.float32, name='temporal_tb')




