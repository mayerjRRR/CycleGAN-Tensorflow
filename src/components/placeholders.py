import tensorflow as tf

frame_sequence_length = 3

class Placeholders:
    def __init__(self, batch_size, image_shape):
        self.init_training_placeholders()
        self.init_real_placeholders(batch_size, image_shape)
        self.init_fake_history_placeholders(batch_size, image_shape)
        self.init_fnet_placeholders(batch_size, image_shape)

    def init_training_placeholders(self):
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.video_training = tf.placeholder(tf.bool, name='video_training')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.global_step = tf.train.get_or_create_global_step(
            graph=None)

    def init_real_placeholders(self, batch_size, image_shape):
        self.image_a = tf.placeholder(tf.float32, [batch_size] + image_shape, name='image_a')
        self.image_b = tf.placeholder(tf.float32, [batch_size] + image_shape, name='image_b')

        self.frames_a = tf.placeholder(tf.float32, [batch_size,frame_sequence_length] + image_shape, name='frames_a')
        self.frames_b = tf.placeholder(tf.float32, [batch_size,frame_sequence_length] + image_shape, name='frames_b')

    def init_fake_history_placeholders(self, batch_size, image_shape):
        self.history_fake_a = tf.placeholder(tf.float32, [batch_size] + image_shape, name='history_fake_a')
        self.history_fake_b = tf.placeholder(tf.float32, [batch_size] + image_shape, name='history_fake_b')

        self.history_fake_warped_frames_a_placeholder = tf.placeholder(tf.float32, [batch_size,frame_sequence_length] + image_shape, name='history_fake_a')
        self.history_fake_warped_frames_b_placeholder = tf.placeholder(tf.float32, [batch_size,frame_sequence_length] + image_shape, name='history_fake_b')

    def init_fnet_placeholders(self, batch_size, image_shape):
        shape = [batch_size] + image_shape
        shape[-1] = shape[-1]*2
        self.fnet_input_placeholder = tf.placeholder(tf.float32, shape, name='fnet_input')

        warp_input_shape = [batch_size] + [frame_sequence_length] + image_shape

        self.image_warp_input = tf.placeholder(tf.float32, warp_input_shape, name='fnet_input')
        self.fake_warp_input = tf.placeholder(tf.float32, warp_input_shape, name='fnet_input')

