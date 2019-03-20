import os
from datetime import datetime

import tensorflow as tf

from src.data_loader import get_training_datasets
from src.cycle_gan import CycleGan
import src.utils.argument_parser as argument_parser
from src.utils.fast_saver import FastSaver
from src.utils.utils import logger, makedirs


def is_video_data(train_A):
    return len(train_A.output_shapes) is 5 and (train_A.output_shapes[1] > 1)


def train(model, train_A, train_B, logdir, learning_rate):
    # TODO: extract into class or method
    next_a = train_A.make_one_shot_iterator().get_next()
    next_b = train_B.make_one_shot_iterator().get_next()
    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    logger.info('Trainable vars:')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)

    var_list_fnet = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
    fnet_loader = tf.train.Saver(var_list_fnet)

    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())
    summary_writer = tf.summary.FileWriter(logdir)

    def init_fn(sess):
        logger.info('Initializing all parameters.')
        sess.run(init_all_op)
        fnet_loader.restore(sess, './fnet/fnet-0')

    logger.info("Starting training session.")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        init_fn(sess)
        summary_writer.add_graph(sess.graph)
        model.savers.load_all(sess)

        if (model.train_videos):
            model.train_on_videos(sess, summary_writer, next_a, next_b,learning_rate)
        else:
            model.train_on_images(sess, summary_writer, next_a, next_b,learning_rate)


def main():
    args, _ = argument_parser.get_train_parser().parse_known_args()

    logger.info('Build datasets:')

    if not args.force_image:
        train_A, train_B = get_training_datasets(args.task, args.image_size, args.batch_size,
                                             dataset_dir=args.dataset_directory)
    else:
        train_A, train_B = get_training_datasets(args.task, args.image_size, args.batch_size,
                                                 dataset_dir=args.dataset_directory, frame_sequence_length=1)

    train_videos = is_video_data(train_A)

    if args.load_model != '':
        model_name = args.load_model
    else:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"{args.task}_{date}"
    logdir = args.log_directory
    makedirs(logdir)
    init_dir = os.path.join(logdir, args.init_model)
    save_dir = os.path.join(logdir, model_name)
    logger.info('Events directory: %s', save_dir)

    logger.info('Build graph:')
    # TODO: extend for hybrid data set
    model = CycleGan(save_dir=save_dir, init_dir=init_dir,image_height=args.image_size, batch_size=args.batch_size, cycle_loss_coeff=args.cycle_loss_coeff,
                     log_step=args.log_step, train_videos=train_videos, train_images=not train_videos)

    train(model, train_A, train_B, save_dir, args.learning_rate)


if __name__ == "__main__":
    main()
