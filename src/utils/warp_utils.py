import tensorflow as tf

from fnet.fnet import fnet


def get_flow(first, second):
    input = tf.concat([first, second], axis=-1)
    with tf.variable_scope('fnet', reuse=tf.AUTO_REUSE):
        flow = fnet(input)
    return tf.image.resize_images(flow, first.shape.as_list()[1:-1])


def warp_to_middle_frame(frame_sequence, flows):
    previous = frame_sequence[:, 0]
    current = frame_sequence[:, 1]
    next = frame_sequence[:, 2]

    backwards_flow, forwards_flow = flows

    previous_warped = tf.contrib.image.dense_image_warp(previous, backwards_flow)
    next_warped = tf.contrib.image.dense_image_warp(next, forwards_flow)

    return tf.stack([previous_warped, current, next_warped], axis=1)


def get_flows_to_middle_frame(frame_sequence):
    frame_sequence = (frame_sequence + 1) / 2
    previous = frame_sequence[:, 0]
    current = frame_sequence[:, 1]
    next = frame_sequence[:, 2]

    backwards_flow = get_flow(previous, current)
    forwards_flow = get_flow(next, current)

    return backwards_flow, forwards_flow


def warp_frame(frame, flow):
    return tf.contrib.image.dense_image_warp(frame, flow)


def pseudorecurrent_inference(generator, current_frame, last_frame, last_result):
    flow = get_flow(last_frame, current_frame)
    last_result_warped = warp_frame(last_result, flow)
    generator_input = tf.concat([current_frame, last_result_warped], axis=-1)
    output = generator(generator_input)
    return output


def apply_inference_on_multiframe(frames, generator):
    frame_list = tf.unstack(frames, axis=1)

    results = []
    last_frame = tf.constant(-1.0, shape=frame_list[0].get_shape().as_list())
    last_result = tf.constant(-1.0, shape=frame_list[0].get_shape().as_list())
    for frame in frame_list:
        result = pseudorecurrent_inference(generator, frame, last_frame, last_result)
        last_frame = frame
        last_result = result
        results.append(result)

    return tf.stack(results, axis=1)

def pingpongify(frames):
    reverse_frames = tf.reverse(frames, axis=[1])
    pingpong_frames = tf.concat([frames, reverse_frames[:,1:]], axis=1)
    return pingpong_frames

def compute_pingpong_difference(pingpong_frames):
    num_frames = pingpong_frames.shape.as_list()[1]
    result_length = num_frames//2
    ping = pingpong_frames[:,:result_length]
    pong = tf.reverse(pingpong_frames[:,-result_length:], axis=[1])
    return tf.abs(ping-pong)