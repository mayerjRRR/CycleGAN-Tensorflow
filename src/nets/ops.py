import tensorflow as tf


def _norm(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            epsilon = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + epsilon)

            channels = input.get_shape()[-1]
            shift = tf.get_variable('shift', shape=[channels],
                                    initializer=tf.zeros_initializer())
            scale = tf.get_variable('scale', shape=[channels],
                                    initializer=tf.random_normal_initializer(1.0, 0.02))

            out = scale * normalized + shift
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.contrib.layers.batch_norm(input,
                                               decay=0.99, center=True,
                                               scale=True, is_training=is_train,
                                               updates_collections=None)
    else:
        out = input

    return out


def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    else:
        return input


def conv2d(input, num_filters, filter_size, stride, reuse=False,
           padding='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]

    weight_initializer = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if padding == 'REFLECT':
        padding = tf.pad(input, _padding_size(input, filter_size, stride), 'REFLECT')
        conv = tf.nn.conv2d(padding, weight_initializer, stride_shape, padding='VALID')
    else:
        assert padding in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, weight_initializer, stride_shape, padding=padding)

    if bias:
        b = tf.get_variable('b', [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0))
        conv = conv + b
    return conv


def _padding_size(tensor, filter_size, stride):
    in_height = int(tensor.get_shape()[1])
    in_width = int(tensor.get_shape()[2])
    if in_height % stride == 0:
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (in_width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return tf.constant([[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])


def conv2d_transpose(input, num_filters, filter_size, stride, reuse,
                     pad='SAME', dtype=tf.float32):
    assert pad == 'SAME'
    batch_size, rows, cols, in_channels = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, in_channels]
    output_shape = [batch_size, int(rows * stride), int(cols * stride), num_filters]

    weight_initializer = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, weight_initializer, output_shape, stride_shape, pad)
    return deconv


def conv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm,
               activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out


def residual(input, num_filters, name, is_train, reuse, norm, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2d(input, num_filters, 3, 1, reuse, pad)
            out = _norm(out, is_train, reuse, norm)
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = conv2d(out, num_filters, 3, 1, reuse, pad)
            out = _norm(out, is_train, reuse, norm)

        return tf.nn.relu(input + out)


def deconv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(input, num_filters, k_size, stride, reuse)
        out = _norm(out, is_train, reuse, norm)
        out = _activation(out, activation)
        return out
