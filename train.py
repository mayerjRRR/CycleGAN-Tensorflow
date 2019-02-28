import os
from datetime import datetime

import tensorflow as tf

from src.data_loader import get_training_datasets
from src.cycle_gan import CycleGan
import src.utils.argument_parser as argument_parser
from src.utils.fast_saver import FastSaver
from src.utils.utils import logger, makedirs


def run(args):
    logger.info('Build datasets:')
    train_A, train_B = get_training_datasets(args.task, args.image_size, args.batch_size,
                                             dataset_dir=args.dataset_directory)

    if args.load_model != '':
        model_name = args.load_model
    else:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"{args.task}_{date}"
    logdir = args.log_directory
    makedirs(logdir)
    logdir = os.path.join(logdir, model_name)
    logger.info('Events directory: %s', logdir)

    logger.info('Build graph:')
    model = CycleGan(logdir, args.image_size, batch_size=args.batch_size, cycle_loss_coeff=args.cycle_loss_coeff,
                     log_step=args.log_step)

    train(model, train_A, train_B, logdir)


def train(model, train_A, train_B, logdir):
    # TODO: extract into class or method
    next_a = train_A.make_one_shot_iterator().get_next()
    next_b = train_B.make_one_shot_iterator().get_next()
    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)
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

        #model.savers.save_all(sess)
        tf.train.write_graph(sess.graph, logdir, 'piff.pbtxt')
        model.train_on_videos(sess, summary_writer, next_a, next_b)


def main():
    args, unparsed = argument_parser.get_train_parser().parse_known_args()
    run(args)


if __name__ == "__main__":
    main()