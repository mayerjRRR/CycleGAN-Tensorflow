import argparse
import cv2


parser = argparse.ArgumentParser(description="Dataset Image Exporter")
parser.add_argument('--input', type=str, default=None, help='Location of the video')
args, _ = parser.parse_known_args()
