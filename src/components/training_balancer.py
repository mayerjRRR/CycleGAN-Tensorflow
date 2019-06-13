from src.components.losses import Losses
import tensorflow as tf


class TrainingBalancer:
    def __init__(self, losses: Losses):
        self.define_balances(losses)

    def define_balances(self, losses: Losses):
        self.spatial_a_balance = get_averaged_tb(losses.D_real_frame_a, losses.D_fake_frame_a)
        self.spatial_b_balance = get_averaged_tb(losses.D_real_frame_b, losses.D_fake_frame_b)
        self.temporal_balance = tf.minimum(get_averaged_tb(losses.D_temp_real_a, losses.D_temp_fake_a),
                                           get_averaged_tb(losses.D_temp_real_b, losses.D_temp_fake_b))


def get_averaged_tb(real_output, fake_output):
    #averager = tf.train.ExponentialMovingAverage(decay=0.99)
    tb_current = get_real_fake_difference(real_output, fake_output)
    #averager.apply([tb_current])
    return tb_current

def get_real_fake_ratio(real_output, fake_output):
    return tf.reduce_mean(real_output) / tf.reduce_mean(fake_output)


def get_real_fake_difference(real_output, fake_output):
    return tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)

class ExponentialMovingAverage:
    def __init__(self, decay=0.99, init_value=0.3):
        self._ema = init_value
        self._decay = 0.99

    def update(self, new_value):
        self._ema = (1 - self._decay) * new_value + self._decay * self._ema

    def evaluate(self):
        return self._ema

    def update_and_evaluate(self, new_value):
        self.update(new_value)
        return self.evaluate()
