import os
import os.path
import tensorflow as tf
from glob import glob as get_all_paths
import cv2

dataset_names = ['trainA', 'trainB', 'testA', 'testB']
image_format_file_ending = 'jpg'

# Use like dis
# a = efficient_data_loader.get_datasets('monet2photo',234)
# iterator = a[0].make_one_shot_iterator()
# next_element = iterator.get_next()
# with tf.Session() as sess:
# cv2.imshow("piff",sess.run(next_element))
# cv2.waitKey(1)

def get_datasets(task_name, image_size) -> [tf.data.Dataset]:
    verify_directory_structure(task_name)
    image_path_tensors = get_image_paths(task_name)
    datasets = build_datasets(image_path_tensors)
    return datasets


def build_datasets(image_path_tensors):
    datasets = []
    for image_path in image_path_tensors:
        dataset = build_dataset(image_path)
        datasets.append(dataset)
    return datasets


def build_dataset(image_path):
    dataset = tf.data.Dataset.from_tensor_slices((image_path))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(64)

    # dataset = dataset.batch(32)
    def load_image(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_normalized = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_normalized = (image_normalized * 2) - 1
        image_resized = tf.image.resize_images(image_normalized, [128, 128])
        return image_resized

    dataset = dataset.map(load_image)
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
        image_path_pattern = os.path.join(data_dir, f"*{image_format_file_ending}")
        task_image_paths = get_all_paths(image_path_pattern)
        image_path_lists.append(task_image_paths)
    return image_path_lists


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
