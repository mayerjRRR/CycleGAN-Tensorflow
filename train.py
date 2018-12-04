import os
from datetime import datetime

import tensorflow as tf

from src.efficient_data_loader import get_training_datasets
from src.cycle_gan import CycleGan
import src.utils.argument_parser as argument_parser
from src.utils.utils import logger, makedirs


class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args):
    logger.info('Build datasets:')
    train_A, train_B = get_training_datasets(args.task, args.image_size, args.batch_size,
                                             dataset_dir=args.dataset_directory)

    logger.info('Build graph:')
    model = CycleGan(args)

    train(args, model, train_A, train_B)


def train(args, model, train_A, train_B):
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
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())
    if args.load_model != '':
        model_name = args.load_model
    else:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"{args.task}_{date}"
    logdir = args.log_directory
    makedirs(logdir)
    logdir = os.path.join(logdir, model_name)
    logger.info('Events directory: %s', logdir)
    summary_writer = tf.summary.FileWriter(logdir)

    def init_fn(sess):
        logger.info('Initializing all parameters.')
        sess.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=model.placeholders.global_step,
                             save_model_secs=300,
                             save_summaries_secs=30)
    if args.train:
        logger.info("Starting training session.")
        with sv.managed_session() as sess:
            model.train(sess, summary_writer, next_a, next_b)


def main():
    args, unparsed = argument_parser.get_train_parser().parse_known_args()
    run(args)


if __name__ == "__main__":
    main()