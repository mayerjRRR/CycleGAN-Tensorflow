import os
from datetime import datetime

import tensorflow as tf

from src.data_loader import get_training_datasets
from src.cycle_gan import CycleGan
import src.utils.argument_parser as argument_parser
from src.utils.utils import get_logger, makedirs

logger = get_logger("main")


def is_video_data(train_A):
    return len(train_A.output_shapes) is 5 and (train_A.output_shapes[1] > 1)


def train(model, train_A, train_B, logdir, learning_rate):
    # TODO: extract into class or method
    next_a = train_A.make_one_shot_iterator().get_next()
    next_b = train_B.make_one_shot_iterator().get_next()
    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    var_list_fnet = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
    fnet_loader = tf.train.Saver(var_list_fnet)

    summary_writer = tf.summary.FileWriter(logdir)

    def init_fn(sess):
        logger.info('Initializing all parameters.')
        sess.run(init_all_op)
        fnet_loader.restore(sess, './fnet/fnet-0')


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        init_fn(sess)
        summary_writer.add_graph(sess.graph)
        model.savers.load_all(sess)

        logger.info(f"Starting {'video' if model.train_videos else 'image'} training.")
        if (model.train_videos):
            model.train_on_videos(sess, summary_writer, next_a, next_b,learning_rate)
        else:
            model.train_on_images(sess, summary_writer, next_a, next_b,learning_rate)


def main():
    training_config = argument_parser.get_training_config()
    logger.info('Building datasets...')
    if not training_config.force_image_training:
        train_A, train_B = get_training_datasets(training_config.task_name, training_config.data_size, training_config.batch_size,
                                             dataset_dir=training_config.dataset_directory, frame_sequence_length=training_config.frame_sequence_length, force_video=training_config.force_video_data)
    else:
        train_A, train_B = get_training_datasets(training_config.task_name, training_config.data_size, training_config.batch_size,
                                                 dataset_dir=training_config.dataset_directory, frame_sequence_length=1, force_video=training_config.force_video_data)

    train_videos = is_video_data(train_A)
    for i in range(training_config.training_runs):
        if training_config.model_directory == '' or training_config.training_runs > 1:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            training_config.model_directory = f"{training_config.task_name}_{date}_{i}-{training_config.training_runs}"
        log_dir = training_config.logging_directory
        makedirs(log_dir)
        training_config.initialization_model = os.path.join(log_dir, training_config.initialization_model)
        training_config.model_directory = os.path.join(log_dir, training_config.model_directory)
        logger.info(f"Checkpoints and Logs will be saved to {training_config.model_directory}")

        logger.info('Building cyclegan:')
        model = CycleGan(training_config=training_config, train_videos=train_videos, train_images=not train_videos)

        train(model, train_A, train_B, training_config.model_directory, training_config.learning_rate)

        tf.reset_default_graph()

if __name__ == "__main__":
    main()
