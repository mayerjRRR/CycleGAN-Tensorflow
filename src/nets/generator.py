import tensorflow as tf
from src.utils.utils import get_logger
from src.nets import ops

logger = get_logger("generator")


class Generator(object):
    def __init__(self, name, is_train, norm='instance', activation='relu', unet=False):
        logger.info(f"Initializing Generator {name}", )
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._num_res_block = 10
        self._reuse = False
        self._unet = unet

    def __call__(self, input, return_code_layer=False):
        with tf.variable_scope(self.name, reuse=self._reuse):
            C1 = ops.conv_block(input, 32, 'c7s1-32', 7, 1, self._is_train,
                                self._reuse, self._norm, self._activation, pad='REFLECT')
            C2 = ops.conv_block(C1, 64, 'd64', 3, 2, self._is_train,
                                self._reuse, self._norm, self._activation)
            C3 = ops.conv_block(C2, 128, 'd128', 3, 2, self._is_train,
                                self._reuse, self._norm, self._activation)

            G = C3
            for i in range(self._num_res_block):
                if i == self._num_res_block // 2 and return_code_layer:
                    return G
                G = ops.residual(G, 128, 'R128_{}'.format(i), self._is_train,
                                 self._reuse, self._norm)
            if self._unet:
                G = ops.deconv_block(tf.concat([G, C3], axis=-1), 64, 'u64', 3, 2, self._is_train,
                                     self._reuse, self._norm, self._activation)
                G = ops.deconv_block(tf.concat([G, C2], axis=-1), 32, 'u32', 3, 2, self._is_train,
                                     self._reuse, self._norm, self._activation)
                G = ops.conv_block(tf.concat([G, C1], axis=-1), 3, 'c7s1-3', 7, 1, self._is_train,
                                   self._reuse, norm=None, activation='tanh', pad='REFLECT')

            else:
                G = ops.deconv_block(G, 64, 'u64', 3, 2, self._is_train,
                                     self._reuse, self._norm, self._activation)
                G = ops.deconv_block(G, 32, 'u32', 3, 2, self._is_train,
                                     self._reuse, self._norm, self._activation)
                G = ops.conv_block(G, 3, 'c7s1-3', 7, 1, self._is_train,
                                   self._reuse, norm=None, activation='tanh', pad='REFLECT')
            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return G
