import argparse


def get_train_parser():

    #TODO: Add identity weight and all the things that are now hardcoded
    parser = argparse.ArgumentParser(description="Training commands")
    parser.add_argument('--task', type=str, default='vidzebra',
                        help='Task name')
    parser.add_argument('--dataset_directory', type=str, default='datasets', help='Location of the training data')
    parser.add_argument('--log_directory', type=str, default='./logs', help='Location that the logs will we stored')
    parser.add_argument('--temp_loss_coeff', type=float, default=1,
                        help='Temporal Discriminator Loss coefficient')
    parser.add_argument('--cycle_loss_coeff', type=float, default=10,
                        help='Cycle Consistency Loss coefficient')
    parser.add_argument('--identity_loss_coeff', type=float, default=5,
                        help='Identity Loss coefficient')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial Learning Rate')
    parser.add_argument('--instance_normalization', default=True, type=bool,
                        help="Use instance norm instead of batch norm")
    parser.add_argument('--log_step', default=100, type=int,
                        help="Tensorboard log frequency")
    parser.add_argument('--save_step', default=500, type=int,
                        help="Model save frequency")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Batch size")
    parser.add_argument('--image_size', default=256, type=int,
                        help="Image size")
    parser.add_argument('--load_model', default='',
                        help='Model path to load and save to (e.g., train_2017-07-07_01-23-45)')
    parser.add_argument('--init_model', default='',
                        help='Model path to initialize model from (e.g., train_2017-07-07_01-23-45)')
    parser.add_argument('--force_image', default=False, type=bool,
                        help='Force image training even with video data')
    return parser

def get_inference_parser():
    parser = argparse.ArgumentParser(description="Inference commands")

    parser.add_argument('--input', type=str, help='Location of the input',default='test_image.jpeg')
    parser.add_argument('--output', type=str, help='Location of the desired output',default='test_output.jpeg')
    parser.add_argument('--forwards', dest='forwards', action='store_true')
    parser.add_argument('--backwards', dest='forwards', action='store_false')
    parser.set_defaults(forwards=True)
    parser.add_argument('--model_dir', type=str,
                        help='Model path to load (e.g., train_2017-07-07_01-23-45)')
    return parser