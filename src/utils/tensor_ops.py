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

def crop_away_borders(input_frame, border_width):
    shape = tf.shape(input_frame)
    new_width = shape[0]-2*border_width
    new_height = shape[1]-2*border_width
    cropped_image = tf.image.resize_image_with_crop_or_pad(
        input_frame,
        new_height,
        new_width
    )
    return cropped_image

def generate_temp_discriminator_input(input_frame):
    layered_frames = layer_frames_in_channels(input_frame)
    cropped_frames = crop_away_borders(layered_frames, 16)
    return cropped_frames