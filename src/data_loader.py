import os
import os.path
import tensorflow as tf
import numpy as np
from glob import glob as get_all_paths
from src.utils.utils import get_logger

from src.video_preprocessor import preprocess_videos
from src.utils.utils import contains_videos

dataset_names = ['trainA', 'trainB']
image_format_file_ending = 'jpg'
video_format_file_ending = 'mp4'
video_index_padding = 1 + 6 + 1

logger = get_logger("data_loader")


def get_training_datasets(task_name, image_size, batch_size, dataset_dir="datasets",frame_sequence_length = 3) -> [tf.data.Dataset]:
    with tf.device('/cpu:0'):
        verify_directory_structure(task_name, dataset_dir)
        image_path_tensors = get_image_paths(task_name, dataset_dir,frame_sequence_length)
        datasets = build_datasets(image_path_tensors, image_size, batch_size)
        return datasets


def build_datasets(image_path_tensors, image_size, batch_size):
    datasets = []
    for image_path in image_path_tensors:
        dataset = build_dataset(image_path, image_size, batch_size)
        datasets.append(dataset)
    return datasets


def build_dataset(image_path, image_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(image_path)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(16)

    def load_image(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_normalized = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        image_normalized = (image_normalized * 2) - 1

        shape = tf.shape(image_normalized)
        min_resolution = tf.minimum(shape[0],shape[1])
        image_cropped = tf.image.resize_image_with_crop_or_pad(
            image_normalized,
            min_resolution,
            min_resolution
        )
        image_resized = tf.image.resize_images(image_cropped, [image_size, image_size])
        return image_resized

    def load_images(filenames):
        return tf.map_fn(load_image, filenames, dtype=tf.float32)

    dataset = dataset.map(load_images)
    dataset = dataset.batch(batch_size)
    return dataset


def get_image_paths(task_name, dataset_dir,frame_sequence_length):
    image_path_lists = get_path_lists(task_name, dataset_dir,frame_sequence_length)
    image_path_tensors = get_path_tensors(image_path_lists)

    return image_path_tensors


def get_path_tensors(image_path_lists):
    image_path_tensors = []
    for path_list in image_path_lists:
        path_tensor = tf.convert_to_tensor(path_list, dtype=tf.string)
        image_path_tensors.append(path_tensor)
    return image_path_tensors


def get_path_lists(task_name, dataset_dir, frame_sequence_length):
    image_path_lists = []
    for dir_name in dataset_names:
        base_dir = os.path.join(dataset_dir, task_name)
        data_dir = os.path.join(base_dir, dir_name)
        is_video_data = contains_videos(data_dir)
        logger.info(f"Training with {'video' if is_video_data else 'image'} data from {data_dir}")
        if is_video_data:
            task_image_paths = get_video_frame_sequences(data_dir, frame_sequence_length)
        else:
            task_image_paths = get_image_frame_sequences(data_dir)
        image_path_lists.append(task_image_paths)
    return image_path_lists

def get_image_frame_sequences(data_dir):
    task_image_paths = np.array([get_path_list(data_dir)]).transpose()
    return task_image_paths


def get_path_list(data_dir):
    image_path_pattern = os.path.join(data_dir, f"*{image_format_file_ending}")
    task_image_paths = get_all_paths(image_path_pattern)
    task_image_paths.sort(key=str.lower)
    return task_image_paths


def verify_directory_structure(task_name, dataset_dir):
    if not os.path.exists(dataset_dir):
        raise Exception("Dataset Directory does not exist!")

    base_dir = os.path.join(dataset_dir, task_name)
    if not os.path.exists(base_dir):
        raise Exception("Task Dataset Directory does not exist!")
    for dataset_name in dataset_names:
        dataset_directory = os.path.join(base_dir, dataset_name)
        if not os.path.exists(dataset_directory):
            raise Exception(f"{dataset_directory} does not exist!")
        if contains_videos(dataset_directory):
            preprocess_videos(dataset_directory)


def get_video_names(task_name):
    image_paths = get_path_list(os.path.join(task_name, 'frames'))
    videos = set([])
    for path in image_paths:
        videos.add(path[:-(video_index_padding + len(image_format_file_ending))])
    return list(videos)


def get_video_frames(video_name):
    all_frames = get_all_paths(video_name + "_*." + image_format_file_ending)
    all_frames.sort(key=str.lower)
    return all_frames


def get_video_frame_sequences(task_name, sequencial_frames):
    video_names = get_video_names(task_name)
    #TODO: make free of transpose
    frame_sequences = [[] for _ in range(sequencial_frames)]
    for video_name in video_names:
        frames = get_video_frames(video_name)
        consecutive_frames = get_consecutive_frames(frames, sequencial_frames)
        frame_sequences = np.concatenate((frame_sequences, consecutive_frames), axis=1)

    if frame_sequences.size == 0:
        frames = get_path_list(os.path.join(task_name, "frames"))
        consecutive_frames = get_consecutive_frames(frames, sequencial_frames)
        frame_sequences = np.concatenate((frame_sequences, consecutive_frames), axis=1)

    return frame_sequences.transpose()


def get_consecutive_frames(frames, num_frames):
    result = []
    for frame_id in range(num_frames):
        result.append(frames[frame_id:len(frames) - num_frames + frame_id + 1])
    return result


