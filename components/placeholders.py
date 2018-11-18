import tensorflow as tf


class Placeholders:
    def __init__(self, batch_size, image_shape):
        self.init_training_placeholders()
        self.init_real_placeholders(batch_size, image_shape)
        self.init_fake_placeholders(image_shape)
        # image_a, image_b = self.augment_data_for_training(image_a, image_b)

    def init_training_placeholders(self):
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.global_step = tf.contrib.framework.get_or_create_global_step(
            graph=None)  # Tensorflow magic global training step index

    def init_real_placeholders(self, batch_size, image_shape):
        self.image_a = tf.placeholder(tf.float32, [batch_size] + image_shape, name='image_a')
        self.image_b = tf.placeholder(tf.float32, [batch_size] + image_shape, name='image_b')

    def init_fake_placeholders(self, image_shape):
        self.history_fake_a_placeholder = tf.placeholder(tf.float32, [None] + image_shape, name='history_fake_a')
        self.history_fake_b_placeholder = tf.placeholder(tf.float32, [None] + image_shape, name='history_fake_b')