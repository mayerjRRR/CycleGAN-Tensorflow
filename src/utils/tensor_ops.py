import tensorflow as tf


def layer_frames_in_channels(input_frames):
    transposed = tf.transpose(input_frames, perm=[0, 2, 3, 1, 4])
    shape = transposed.get_shape().as_list()
    shape[-2] *= shape[-1]
    shape.pop();
    return tf.reshape(transposed, shape);

def extract_frames_from_channels(input_frame):
    shape = input_frame.get_shape().as_list()
    #assuming 3 color channels
    frames_in_sequence = int(shape[-1]/3)
    new_shape = shape[0:-1]+[frames_in_sequence,3]
    reshaped = tf.reshape(input_frame, new_shape);
    return tf.transpose(reshaped, perm=[0, 3, 1, 2, 4])