import cv2
import os

from src.utils.utils import get_all_video_paths
from src.utils.utils import get_logger

logger = get_logger("video_preprocessor")

frame_directory_name = 'frames'


def preprocess_videos(path):
    frame_dir = os.path.join(path, frame_directory_name)
    video_paths = get_all_video_paths(path)

    if os.path.exists(frame_dir):
        logger.info("Frame directory already exists, no preprocessing needed!")
        return
    else:
        os.makedirs(frame_dir)

    logger.info(f"Preprocessing {len(video_paths)} videos.")
    for path in video_paths:
        logger.info(f"Preprocessing {path}...")
        extract_video_frames(path, frame_dir)


def extract_video_frames(video_path, frame_directory):
    videoCapture = cv2.VideoCapture(video_path)
    if not videoCapture.isOpened():
        logger.info('Error opening video {}'.format(video_path))
        return

    ret, frame = videoCapture.read()
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        logger.info('Wrong image {} with shape {}'.format(video_path, frame.shape))
        return

    image_format = ".jpg"
    directory_name, file_name = os.path.split(video_path)
    video_name = os.path.splitext(file_name)[0]
    if frame_directory is None:
        frame_directory = os.path.join(directory_name, frame_directory_name)

    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)
    frame_index = 0
    while ret:
        frame_file_name = os.path.join(frame_directory, video_name + "_" + str(frame_index).zfill(6) + image_format)
        cv2.imwrite(frame_file_name, frame)
        ret, frame = videoCapture.read()
        frame_index += 1

    logger.info(f"Extracted {frame_index} frames.")
    videoCapture.release()
