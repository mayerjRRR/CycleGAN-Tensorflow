import argparse


def get_train_parser():

    parser = argparse.ArgumentParser(description="Training commands")
    parser.add_argument('--task', type=str, default='smoke',
                        help='Task name')
    parser.add_argument('--dataset_directory', type=str, default='datasets', help='Location of the training data')
    parser.add_argument('--log_directory', type=str, default='./logs', help='Location that the logs will we stored')
    parser.add_argument('--temp_loss_coeff', type=float, default=0.5,
                        help='Temporal Discriminator Loss coefficient')
    parser.add_argument('--cycle_loss_coeff', type=float, default=10,
                        help='Cycle Consistency Loss coefficient')
    parser.add_argument('--identity_loss_coeff', type=float, default=50,
                        help='Identity Loss coefficient')
    parser.add_argument('--pingpong_loss_coeff', type=float, default=1.0,
                        help='Identity Loss coefficient')
    parser.add_argument('--code_loss_coeff', type=float, default=1.0,
                        help='Identity Loss coefficient')
    parser.add_argument('--identity_loss_fadeout', type=bool, default=True,
                        help='Whether the identity loss should fade out.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial Learning Rate')
    #TODO: Remove, Batch norm sucks
    parser.add_argument('--instance_normalization', default=True, type=bool,
                        help="Use instance norm instead of batch norm")
    parser.add_argument('--log_step', default=100, type=int,
                        help="Tensorboard log frequency")
    parser.add_argument('--save_step', default=500, type=int,
                        help="Model save frequency")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Batch size")
    parser.add_argument('--training_size', default=128, type=int,
                        help="Resolution of the images for training")
    parser.add_argument('--data_size', default=512, type=int,
                        help="Resolution of the data")
    parser.add_argument('--load_model', default='',
                        help='Model path to load and save to (e.g., train_2017-07-07_01-23-45)')
    parser.add_argument('--init_model', default='',
                        help='Model path to initialize model from (e.g., train_2017-07-07_01-23-45)')
    parser.add_argument('--force_image', default=False, type=bool,
                        help='Force image training even with video data')
    parser.add_argument('--force_video', default=False, type=bool,
                        help='Force video training even if videos files not present, frames directory must exist')
    parser.add_argument('--frame_seq_length', default=4, type=int,
                        help="Length of the frame sequence for training.")
    return parser

class TrainingConfig:
    def __init__(self, args):
        self.task_name = args.task
        self.dataset_directory = args.dataset_directory
        self.logging_directory = args.log_directory

        self.learning_rate = args.learning_rate
        self.frame_sequence_length = args.frame_seq_length
        self.batch_size = args.batch_size
        self.training_size = args.training_size
        self.data_size = args.data_size
        self.use_instance_normalization = args.instance_normalization
        self.temporal_loss_coefficient = args.temp_loss_coeff
        self.cycle_loss_coefficient = args.cycle_loss_coeff
        self.identity_loss_coefficient = args.identity_loss_coeff
        self.code_loss_coefficient = args.code_loss_coeff
        self.pingpong_loss_coefficient = args.pingpong_loss_coeff
        self.fadeout_identity_loss = args.identity_loss_fadeout
        self.force_image_training = args.force_image
        self.force_video_data = args.force_video

        self.logging_frequency = args.log_step
        self.saving_frequency = args.save_step

        self.model_directory = args.load_model
        self.initialization_model = args.init_model

def get_training_config():
    args, _ = get_train_parser().parse_known_args()
    training_config = TrainingConfig(args)
    return training_config

def get_inference_parser():
    parser = argparse.ArgumentParser(description="Inference commands")

    parser.add_argument('--input', type=str, help='Location of the input',default='results/test_image.jpeg')
    parser.add_argument('--output', type=str, help='Location of the desired output',default='results/test_output.jpeg')
    parser.add_argument('--forwards', dest='forwards', action='store_true')
    parser.add_argument('--backwards', dest='forwards', action='store_false')
    parser.set_defaults(forwards=True)
    parser.add_argument('--side_by_side', dest='with_old', action='store_true')
    parser.set_defaults(with_old=False)
    parser.add_argument('--model_dir', type=str,
                        help='Model path to load (e.g., train_2017-07-07_01-23-45)')
    return parser