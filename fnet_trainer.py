from time import gmtime, strftime

import tensorflow as tf

from fnet.fnet import fnet
from src.data_loader import get_training_datasets


def train(image_size=256, batch_size=8, dataset_directory="datasets", task="vidzebra"):
    fnet_input = get_input(dataset_directory, task, batch_size, image_size)
    optical_flow_estimate = get_flow_estimate(fnet_input)
    first_frame, first_frame_warped, second_frame = warp_first_frame(fnet_input, optical_flow_estimate)
    l1_warp_loss, warp_difference_bw = define_loss(first_frame_warped, second_frame)

    saver = get_saver()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(l1_warp_loss)

    summary_op = create_summary(first_frame, first_frame_warped, l1_warp_loss, second_frame, warp_difference_bw)
    summary_writer = tf.summary.FileWriter('./fnet/log/' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

    # Define the initialization operation
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        initialize(init_op, saver, sess, summary_writer)
        iteration = 0;
        while True:

            to_run = [optimizer]
            if iteration % 100 == 0:
                to_run += [summary_op]
            result = sess.run(to_run)

            if iteration % 100 == 0:
                write_summary(iteration, result, summary_writer)
            if iteration % 550 == 0:
                save_model(iteration, saver, sess)
            iteration += 1


def get_input(dataset_directory, task, batch_size, image_size):
    _, zebra_frames = get_training_datasets(task, image_size, batch_size,
                                            dataset_dir=dataset_directory, frame_sequence_length=2)
    frame_pair = zebra_frames.make_one_shot_iterator().get_next()
    fnet_input = reshape_for_fnet(frame_pair, image_size, batch_size)
    return fnet_input


def reshape_for_fnet(frame_pair, image_size, batch_size):
    reshaped = tf.transpose(frame_pair, perm=[0, 2, 3, 1, 4])
    fnet_input = tf.reshape(reshaped, [batch_size, image_size, image_size, 6]);
    return fnet_input


def get_flow_estimate(fnet_input):
    with tf.variable_scope('fnet'):
        optical_flow_estimate = fnet(fnet_input, reuse=False)
    return optical_flow_estimate


def warp_first_frame(fnet_input, optical_flow_estimate):
    first_frame, second_frame = tf.split(fnet_input, num_or_size_splits=2, axis=-1)
    first_frame_warped = tf.contrib.image.dense_image_warp(first_frame, optical_flow_estimate)
    return first_frame, first_frame_warped, second_frame


def define_loss(first_frame_warped, second_frame):
    warp_diff_abs = tf.abs(second_frame - first_frame_warped)
    l1_warp_loss = tf.reduce_mean(warp_diff_abs)
    warp_difference_bw = tf.reduce_sum(warp_diff_abs, axis=[3], keep_dims=True)
    return l1_warp_loss, warp_difference_bw


def get_saver():
    var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
    saver = tf.train.Saver(var_list)
    return saver


def create_summary(first_frame, first_frame_warped, l1_warp_loss, second_frame, warp_difference_bw):
    tf.summary.scalar('L1-loss', l1_warp_loss)
    tf.summary.image('rrr/First', first_frame[0:1])
    tf.summary.image('rrr/Second', second_frame[0:1])
    tf.summary.image('rrr/Warped', first_frame_warped[0:1])
    tf.summary.image('rrr/Error', warp_difference_bw[0:1])
    summary_op = tf.summary.merge_all()
    return summary_op


def initialize(init_op, saver, sess, summary_writer):
    sess.run(init_op)
    saver.restore(sess, './fnet/fnet-0')
    summary_writer.add_graph(sess.graph)


def write_summary(iteration, result, summary_writer):
    summary_writer.add_summary(result[-1], iteration)
    summary_writer.flush()
    print("writing log")


def save_model(iteration, saver, sess):
    print("saving...")
    saver.save(sess, './fnet/trained/fnet', global_step=iteration)


if __name__ == '__main__':
    train()
