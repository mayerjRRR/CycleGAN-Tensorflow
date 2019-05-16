import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
import tensorflow as tf

#'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'
def get_vgg_activations(input, layers=['vgg_19/conv2/conv2_1','vgg_19/conv3/conv3_1','vgg_19/conv4/conv4_1']):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg_19(input)
    activations = [end_points[layer] for layer in layers]
    return activations

#modified vgg_19 from tf.slim.nets.vgg to be reusable
def vgg_19(inputs,
           scope='vgg_19', reuse=tf.AUTO_REUSE):
    """Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            net = slim.repeat(
                inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points
