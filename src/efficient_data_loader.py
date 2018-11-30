import os
import os.path
import tensorflow as tf
import numpy as np
from glob import glob as get_all_paths

dataset_names = ['trainA', 'trainB', 'testA', 'testB']
image_format_file_ending = 'jpg'
video_format_file_ending = 'mp4'
video_index_padding = 1+6+1


def get_datasets(task_name, image_size, batch_size) -> [tf.data.Dataset]:
    with tf.device('/cpu:0'):
        verify_directory_structure(task_name)
        image_path_tensors = get_image_paths(task_name)
        datasets = build_datasets(image_path_tensors, image_size, batch_size)
        return datasets


def build_datasets(image_path_tensors, image_size, batch_size):
    datasets = []
    for image_path in image_path_tensors:
        print(f"Shape of Tensor: {tf.shape(image_path)}")
        dataset = build_dataset(image_path, image_size, batch_size)
        datasets.append(dataset)
    return datasets


def build_dataset(image_path, image_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(image_path)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(64)

    def load_images(filenames):
        return tf.map_fn(load_image, filenames, dtype=tf.float32)

    def load_image(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_normalized = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        image_normalized = (image_normalized * 2) - 1
        image_resized = tf.image.resize_images(image_normalized, [image_size, image_size])
        return image_resized

    dataset = dataset.map(load_images)
    dataset = dataset.batch(batch_size)
    return dataset


def get_image_paths(task_name):
    # TODO: Replace with multiframe paths
    image_path_lists = get_path_lists(task_name)
    image_path_tensors = get_path_tensors(image_path_lists)

    return image_path_tensors


def get_path_tensors(image_path_lists):
    image_path_tensors = []
    for path_list in image_path_lists:
        path_tensor = tf.convert_to_tensor(path_list, dtype=tf.string)
        image_path_tensors.append(path_tensor)
    return image_path_tensors


def get_path_lists(task_name):
    image_path_lists = []
    for dir_name in dataset_names:
        base_dir = os.path.join('datasets', task_name)
        data_dir = os.path.join(base_dir, dir_name)
        task_image_paths = get_frame_sequences(data_dir,3)
        #task_image_paths = get_path_list(data_dir)
        image_path_lists.append(task_image_paths)
    return image_path_lists


def get_path_list(data_dir):
    image_path_pattern = os.path.join(data_dir, f"*{image_format_file_ending}")
    task_image_paths = get_all_paths(image_path_pattern)
    return task_image_paths


def verify_directory_structure(task_name):
    if not os.path.exists('datasets'):
        raise Exception("Dataset Directory does not exist!")

    base_dir = os.path.join('datasets', task_name)
    if not os.path.exists(base_dir):
        raise Exception("Task Dataset Directory does not exist!")
    for dataset_name in dataset_names:
        dataset_directory = os.path.join(base_dir, dataset_name)
        if not os.path.exists(dataset_directory):
            raise Exception(f"{dataset_directory} does not exist!")


def get_video_names(task_name):
    image_paths = get_path_list(os.path.join(task_name, 'frames'))
    videos = set([])
    for path in image_paths:
        videos.add(path[:-(video_index_padding+len(image_format_file_ending))])
    return list(videos)


def get_video_frames(video_name):
    return get_all_paths(video_name + "_*." + image_format_file_ending)

def get_frame_sequences(task_name, sequencial_frames):
    video_names = get_video_names(task_name)
    frame_sequences = [[] for _ in range(sequencial_frames)]
    for video_name in video_names:
        frames = get_video_frames(video_name)
        consecutive_frames = get_consecutive_frames(frames, sequencial_frames)
        frame_sequences = np.concatenate((frame_sequences, consecutive_frames), axis=1)
    return frame_sequences


def get_consecutive_frames(frames, num_frames):
    result = []
    for frame_id in range(num_frames):
        result.append(frames[frame_id:len(frames)-num_frames+frame_id+1])
    return result
