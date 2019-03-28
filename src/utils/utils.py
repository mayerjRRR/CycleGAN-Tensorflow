import os
import logging
from glob import glob

checkpoint_directory = "logs"

def get_logger(name=None):
    logging.info(f"Start {name}")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_creation_date(path):
    return os.stat(path).st_ctime


def get_subdirs(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def get_latest_model():
    checkpoint_dirs = get_subdirs(checkpoint_directory)
    checkpoint_dirs.sort(key=lambda dir: get_creation_date(dir), reverse=True)
    return checkpoint_dirs[0]


def get_all_video_paths(path):
    video_paths = glob(os.path.join(path, '*.mp4'))
    return video_paths


def contains_videos(path):
    video_paths = get_all_video_paths(path)
    return not(not video_paths)