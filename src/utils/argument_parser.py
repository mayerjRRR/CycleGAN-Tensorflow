import argparse
import os


def get_train_parser():
    parser = argparse.ArgumentParser(description="Training commands")
    parser.add_argument('-n', '--run_name', type=str, default=None,
                        help='Short decription of the run to be used in the save files name.')
    parser.add_argument('--task', type=str, default='smoke',
                        help='Name of the task, also name of the directory wherer the training dataset is stored.')
    parser.add_argument('--dataset_directory', type=str, default='datasets', help='Location of the training data main directory.')
    parser.add_argument('--training_iter', default=100, type=int,
                        help="Total number of training iterations.")
    parser.add_argument('--decay_iter', default=20, type=int,
                        help="Number of  iterations with learning rate decay.")
    parser.add_argument('--cross_entropy_gan_loss', type=bool, default=False,
                        help='Use crossentropy, if false, use MSE. Crossentropy broken atm.')
    parser.add_argument('--unet', type=bool, default=False, help='Use Unet architecture instead of Johnson.')
    parser.add_argument('--log_directory', type=str, default='./logs', help='Location that all logs will be stored to.')
    parser.add_argument('--temp_loss_coeff', type=float, default=0.5,
                        help='Temporal discriminator loss coefficient.')
    parser.add_argument('--cycle_loss_coeff', type=float, default=10,
                        help='Cycle consistency loss coefficient.')
    parser.add_argument('--identity_loss_coeff', type=float, default=10,
                        help='Identity loss coefficient.')
    parser.add_argument('--pingpong_loss_coeff', type=float, default=100.0,
                        help='Ping-Pong loss coefficient.')
    parser.add_argument('--code_loss_coeff', type=float, default=1.0,
                        help='Latent space consistency loss coefficient.')
    parser.add_argument('--style_loss_coeff', type=float, default=100000.0,
                        help='Discriminator style loss coefficient.')
    parser.add_argument('--vgg_loss_coeff', type=float, default=1.0,
                        help='VGG-19 style loss coefficient.')
    parser.add_argument('--identity_loss_fadeout', type=bool, default=True,
                        help='Whether the identity loss should fade out.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate.')
    # TODO: Remove, Batch norm sucks
    parser.add_argument('--instance_normalization', default=True, type=bool,
                        help="Use instance norm instead of batch norm.")
    parser.add_argument('--log_step', default=100, type=int,
                        help="TensorBoard logging frequency in training iteration.")
    parser.add_argument('--save_step', default=500, type=int,
                        help="Model saving frequency in training iteration.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Batch size, leave at 1 unless you have some kind of future alien GPU.")
    parser.add_argument('--training_size', default=256, type=int,
                        help="Resolution of the frames for training. The training images are randomly cropped from the raw frames with resolution <data_size>.")
    parser.add_argument('--data_size', default=288, type=int,
                        help="Resolution of the frames as provided by the data_loader. The frames are later cropped down to <training_size>")
    parser.add_argument('--load_model', default='',
                        help='Model path to load and save to (e.g., train_2017-07-07_01-23-45)')
    parser.add_argument('--init_model', default='',
                        help='Model path to initialize model from (e.g., train_2017-07-07_01-23-45). The model is not saved to this location and does not load the global training step.')
    parser.add_argument('--force_image', default=False, type=bool,
                        help='Force image training even with video data present.')
    parser.add_argument('--force_video', default=True, type=bool,
                        help='Force video training even if videos files not present, but frames directory must exist!')
    parser.add_argument('--frame_seq_length', default=4, type=int,
                        help="Length of the frame sequence for training, used for Ping-Pong loss.")
    parser.add_argument('--training_runs', default=1, type=int,
                        help="Number of training runs to run in sequence. Use to loop training with identical settings for comparison reasons.")
    parser.add_argument('--tb_threshold', type=float, default=0.4,
                        help='Threshold of the average training balancer value to turn off discriminator training.')
    return parser


class TrainingConfig:
    def __init__(self, args):
        self.run_name = args.run_name
        self.task_name = args.task
        self.dataset_directory = args.dataset_directory
        self.logging_directory = args.log_directory
        self.training_iterations = args.training_iter
        self.decay_iterations = args.decay_iter

        self.learning_rate = args.learning_rate
        self.frame_sequence_length = args.frame_seq_length
        self.batch_size = args.batch_size
        self.training_size = args.training_size
        self.data_size = args.data_size
        self.use_instance_normalization = args.instance_normalization
        self.use_crossentropy_loss = args.cross_entropy_gan_loss
        self.use_unet = args.unet
        self.temporal_loss_coefficient = args.temp_loss_coeff
        self.cycle_loss_coefficient = args.cycle_loss_coeff
        self.identity_loss_coefficient = args.identity_loss_coeff
        self.code_loss_coefficient = args.code_loss_coeff
        self.style_loss_coefficient = args.style_loss_coeff
        self.vgg_loss_coefficient = args.vgg_loss_coeff
        self.pingpong_loss_coefficient = args.pingpong_loss_coeff
        self.fadeout_identity_loss = args.identity_loss_fadeout
        self.force_image_training = args.force_image
        self.force_video_data = args.force_video

        self.logging_frequency = args.log_step
        self.saving_frequency = args.save_step

        self.model_directory = args.load_model
        self.initialization_model = args.init_model

        self.training_runs = args.training_runs
        self.training_balancer_threshold = args.tb_threshold

    def __str__(self):
        dictionary = self.__dict__
        result = ""
        for key in dictionary:
            result += key + ": " + str(dictionary[key]) + "  " + os.linesep
        return result


def get_training_config():
    args, _ = get_train_parser().parse_known_args()
    training_config = TrainingConfig(args)
    return training_config


def get_inference_parser():
    parser = argparse.ArgumentParser(description="Inference commands")

    parser.add_argument('--input', type=str, help='Location of the input', default='results/smoke_input.jpg')
    parser.add_argument('--output', type=str, help='Location of the desired output', default='results/smoke_output.jpg')
    parser.add_argument('--forwards', dest='forwards', action='store_true')
    parser.add_argument('--backwards', dest='forwards', action='store_false')
    parser.set_defaults(forwards=True)
    parser.add_argument('--side_by_side', dest='with_old', action='store_true')
    parser.set_defaults(with_old=False)
    parser.add_argument('--model_dir', type=str,
                        help='Model path to load (e.g., train_2017-07-07_01-23-45).')
    parser.add_argument('--model_super_dir', type=str,
                        help='Path to directrory containing lots of model directories (e.g., ./logs). Use this if you want to test lots of models at once.')
    parser.add_argument('--width', default=None, type=int,
                        help="Width of output.")
    parser.add_argument('--height', default=None, type=int,
                        help="Height of output.")
    parser.add_argument('--unet', type=bool, default=False, help='Use Unet architecture instead of Johnson.')
    parser.add_argument('--no_temp', type=bool, default=False, help='No frame-recurrent generator aka standard cycleGAN generator.')

    return parser
