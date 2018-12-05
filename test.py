from src.utils import argument_parser
import os
import cv2
from src.cycle_gan import CycleGan
import tensorflow as tf
from src.utils.fast_saver import FastSaver
from src.utils.image_utils import load_float_image, save_float_image

#TODO: Support Video Inference
#TODO; Support Image Directory Inference
#TODO: Support Backwards Inference
#TODO: Support inference on lastest model directory

def run(args):
    #parseArgs
    forwards, input, model_dir, output = parse_arguments(args)

    #load input file

    directory_name, file_name = os.path.split(input)
    input_name, file_ending = os.path.splitext(file_name)

    if file_ending == ".mp4":
        processVideo()
    elif file_ending == ".jpg" or file_ending == ".png" or file_ending == ".jpeg":
        processSingleImage(input, output, forwards, model_dir)

    print("Done!")
    #cleanup


def parse_arguments(args):
    input = args.input
    output = args.output
    forwards = args.forwards
    model_dir = args.model_dir
    return forwards, input, model_dir, output


def processVideo():
        pass

def processSingleImage(input, output, forwards, model_dir):

    input_image = load_float_image(input)

    #determine resolution
    height, width, _ = input_image.shape
    #TODO: Support Non-Square Inference
    input_image = cv2.resize(input_image, (height, height))

    inference_machine = InferenceMachine(height, width, model_dir)

    ab_image = inference_machine.forward_inference(input_image)

    save_float_image(ab_image, output)

class InferenceMachine():
    def __init__(self, height, width, model_dir):
        self.model = CycleGan(image_size=height, batch_size=1)
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

        self.sv = tf.train.Supervisor(is_chief=True,
                                 logdir=logdir,
                                 saver=saver,
                                 summary_op=None,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 summary_writer=summary_writer,
                                 ready_op=tf.report_uninitialized_variables(variables_to_save),
                                 global_step=self.model.placeholders.global_step,
                                 save_model_secs=300,
                                 save_summaries_secs=30)


    def forward_inference(self, input_image):
        with self.sv.managed_session() as sess:
            result = sess.run(self.model.images.image_ab, feed_dict={self.model.placeholders.image_a: [input_image],
                                                                   self.model.placeholders.is_train: False})[0]
        return result

    def backward_inference(self, input_image):
        with self.sv.managed_session() as sess:
            result = sess.run(self.model.images.image_ba, feed_dict={self.model.placeholders.image_b: [input_image],
                                                                   self.model.placeholders.is_train: False})[0]
        return result



def main():
    args, unparsed = argument_parser.get_inference_parser().parse_known_args()
    run(args)


if __name__ == "__main__":
    main()