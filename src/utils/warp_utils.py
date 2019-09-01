import tensorflow as tf

from fnet.fnet import fnet


def get_flow(first, second):
    with tf.name_scope("get_flow"):
        input = tf.concat([first, second], axis=-1)
        with tf.variable_scope('fnet', reuse=tf.AUTO_REUSE):
            flow = fnet(input)
        result = tf.image.resize_images(flow, first.shape.as_list()[1:-1])
    return result


def stack_triplets_in_batch(frames):
    num_frames = frames.shape.as_list()[1]
    num_triplets = num_frames // 3

    triplet_list = []
    frame_list = tf.unstack(frames, axis=1)
    for i in range(num_triplets):
        triplet_list.append(tf.stack(frame_list[3 * i:3 * (i + 1)], axis=1))

    result = tf.concat(triplet_list, axis=0)
    return result


def warp_to_middle_frame(frame_sequence, flows):
    with tf.name_scope("warp_to_middle_frame"):
        previous = frame_sequence[:, -3]
        current = frame_sequence[:, -2]
        next = frame_sequence[:, -1]

        backwards_flow, forwards_flow = flows

        previous_warped = tf.contrib.image.dense_image_warp(previous, backwards_flow)
        next_warped = tf.contrib.image.dense_image_warp(next, forwards_flow)
        result = tf.stack([previous_warped, current, next_warped], axis=1)
    return result


def get_flows_to_middle_frame(frame_sequence):
    with tf.name_scope("get_flows_to_middle_frame"):
        frame_sequence = (frame_sequence + 1) / 2
        previous = frame_sequence[:, -3]
        current = frame_sequence[:, -2]
        next = frame_sequence[:, -1]

        backwards_flow = get_flow(previous, current)
        forwards_flow = get_flow(next, current)

    return backwards_flow, forwards_flow


def warp_frame(frame, flow):
    return tf.contrib.image.dense_image_warp(frame, flow)


def get_fake_generator_input(image):
    return tf.concat([image, image], axis=-1)


def recurrent_inference(generator, current_frame, last_frame, last_result):
    with tf.name_scope("recurrent_inference"):
        flow = get_flow(last_frame, current_frame)
        last_result_warped = warp_frame(last_result, flow)
        generator_input = tf.concat([current_frame, last_result_warped], axis=-1)
        output = generator(generator_input)
    return output


def apply_inference_on_multiframe(frames, generator):
    with tf.name_scope("apply_inference_on_multiframe"):
        frame_list = tf.unstack(frames, axis=1)

        results = []
        last_frame = tf.constant(-1.0, shape=frame_list[0].get_shape().as_list())
        last_result = tf.constant(-1.0, shape=frame_list[0].get_shape().as_list())
        for frame in frame_list:
            result = recurrent_inference(generator, frame, last_frame, last_result)
            last_frame = frame
            last_result = result
            results.append(result)
        result = tf.stack(results, axis=1)
    return result


def pingpongify(frames):
    with tf.name_scope("pingpongify"):
        reverse_frames = tf.reverse(frames, axis=[1])
        pingpong_frames = tf.concat([frames, reverse_frames[:, 1:]], axis=1)
    return pingpong_frames


def compute_pingpong_difference(pingpong_frames):
    ping, pong = unpingpongify(pingpong_frames)
    # remove first frame since inference incomplete
    return tf.abs(ping[:, 1:] - pong[:, 1:])


def unpingpongify(pingpong_frames):
    with tf.name_scope("unpingpongify"):
        num_frames = pingpong_frames.shape.as_list()[1]
        result_length = (num_frames // 2) + 1
        ping = pingpong_frames[:, :result_length]
        pong = tf.reverse(pingpong_frames[:, -result_length:], axis=[1])
    return ping, pong


def get_gram_matrix(input_tensor):
    # shape = batch_size, hight, width, channels
    batch_size, height, width, channels = input_tensor.shape.as_list()
    size = height * width * channels
    feature_vector = tf.reshape(input_tensor, (-1, height * width, channels))
    feature_vector_transposed = tf.transpose(feature_vector, perm=[0, 2, 1])
    gram_matrix = tf.matmul(feature_vector_transposed, feature_vector) / size
    return gram_matrix
