import tensorflow as tf
from src.utils.utils import get_logger
from src.nets import ops

logger = get_logger("generator")

class Generator(object):
    def __init__(self, name, is_train, norm='instance', activation='relu'):
        logger.info(f"Initializing Generator {name}", )
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._num_res_block = 8
        self._reuse = False

    def __call__(self, input, return_code_layer=False):
        with tf.variable_scope(self.name, reuse=self._reuse):
            G = ops.conv_block(input, 32, 'c7s1-32', 7, 1, self._is_train,
                               self._reuse, self._norm, self._activation, pad='REFLECT')
            G = ops.conv_block(G, 64, 'd64', 3, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            G = ops.conv_block(G, 128, 'd128', 3, 2, self._is_train,
                               self._reuse, self._norm, self._activation)
            for i in range(self._num_res_block):
                if i == self._num_res_block // 2 and return_code_layer:
                    return G
                G = ops.residual(G, 128, 'R128_{}'.format(i), self._is_train,
                                 self._reuse, self._norm)

            G = ops.deconv_block(G, 64, 'u64', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)
            G = ops.deconv_block(G, 32, 'u32', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)
            G = ops.conv_block(G, 3, 'c7s1-3', 7, 1, self._is_train,
                               self._reuse, norm=None, activation='tanh', pad='REFLECT')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return G


