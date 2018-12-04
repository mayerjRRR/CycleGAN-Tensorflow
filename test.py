# logger.info("Starting testing session.")
# with sv.managed_session() as sess:
#    base_dir = os.path.join('results', model_name)
#     makedirs(base_dir)
#    model.test(sess, test_A, test_B, base_dir)
from src.utils import argument_parser
import numpy as np
import os
import cv2
from src.cycle_gan import CycleGan
import tensorflow as tf
from src.utils.fast_saver import FastSaver


def run(args):
    #parseArgs
    input = args.input
    output = args.output
    forwards = args.forwards
    model_dir = args.model_dir

    #load input file
    #support jpeg and mp4

    directory_name, file_name = os.path.split(input)
    input_name, file_ending = os.path.splitext(file_name)

    print(file_ending)

    if file_ending == ".mp4":
        processVideo()
    elif file_ending == ".jpg" or file_ending == ".png":
        processImage(input, output, forwards, model_dir)

    print("done")
    #determine resolution
    #build cyclegan with correct resolution
    #do inferenece
    #cleanup

def processVideo():
        pass

def processImage(input, output, forwards, model_dir):
    print("sdgsgsd")
    image = cv2.imread(input, 0)
    print(image.shape)
    height, width = image.shape
    image = cv2.resize(image, (height, height))
    cycleGan = CycleGan(image_size=height, batch_size=1)

    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)

    logdir = model_dir
    summary_writer = tf.summary.FileWriter(logdir)

    def init_fn(sess):
        sess.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=cycleGan.placeholders.global_step,
                             save_model_secs=300,
                             save_summaries_secs=30)
    with sv.managed_session() as sess:
        img = sess.run(cycleGan.images.image_ab, feed_dict={cycleGan.placeholders.image_a: image,
                                                      cycleGan.placeholders.is_train: False})
        cv2.imshow("ll", img)
        cv2.waitKey(0)


def main():
    args, unparsed = argument_parser.get_inference_parser().parse_known_args()
    run(args)


if __name__ == "__main__":
    main()