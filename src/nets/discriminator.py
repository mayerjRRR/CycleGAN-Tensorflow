import tensorflow as tf
from src.utils.utils import get_logger
from src.nets import ops

logger = get_logger("discriminator")

class Discriminator(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky'):
        logger.info(f"Initializing Discriminator {name}...")
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._reuse = False

    def __call__(self, input, return_layer_activations=False):
        layer_activations = []
        with tf.variable_scope(self.name, reuse=self._reuse):
            D = ops.conv_block(input, 64, 'C64', 4, 2, self._is_train,
                               self._reuse, norm=None, activation=self._activation)
            layer_activations.append(D)
            D = ops.conv_block(D, 128, 'C128', 4, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            layer_activations.append(D)
            D = ops.conv_block(D, 256, 'C256', 4, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            layer_activations.append(D)
            D = ops.conv_block(D, 512, 'C512', 4, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            layer_activations.append(D)
            if return_layer_activations:
                return layer_activations
            D = ops.conv_block(D, 1, 'C1', 4, 1, self._is_train,
                               self._reuse, norm=None, activation=None, bias=True)
            D = tf.reduce_mean(D, axis=[1,2,3])

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return D
