import argparse
from src import cycle_gan

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-t', '--train', default=True, type=bool,
                    help="Training mode")
parser.add_argument('--task', type=str, default='apple2orange',
                    help='Task name')
parser.add_argument('--cycle_loss_coeff', type=float, default=10,
                    help='Cycle Consistency Loss coefficient')
parser.add_argument('--instance_normalization', default=True, type=bool,
                    help="Use instance norm instead of batch norm")
parser.add_argument('--log_step', default=100, type=int,
                    help="Tensorboard log frequency")
parser.add_argument('--batch_size', default=1, type=int,
                    help="Batch size")
parser.add_argument('--image_size', default=128, type=int,
                    help="Image size")
parser.add_argument('--load_model', default='',
                    help='Model path to load (e.g., train_2017-07-07_01-23-45)')

args, _ = parser.parse_known_args()
cg = cycle_gan.CycleGan()