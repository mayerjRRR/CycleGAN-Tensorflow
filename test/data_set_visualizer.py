from src.data_loader import get_training_datasets
import tensorflow as tf
import numpy as np
import cv2
import argparse

from src.utils.utils import get_subdir_names


def create_dataset_image(dataset, num_images, dataset_dir="datasets", resolution=512):
    a_dataset, b_dataset = get_training_datasets(dataset, resolution, 4, dataset_dir, 3, force_video=True)
    a_batch = a_dataset.make_one_shot_iterator().get_next()
    b_batch = b_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        for i in range(num_images):
            a_value = sess.run(a_batch)
            b_value = sess.run(b_batch)
            a_image = np.reshape(a_value, (2, resolution * 6, resolution, 3))
            a_image = np.concatenate((a_image[0], a_image[1]), axis=1)
            b_image = np.reshape(b_value, (2, resolution * 6, resolution, 3))
            b_image = np.concatenate((b_image[0], b_image[1]), axis=1)
            final = np.concatenate((a_image, b_image), axis=1)
            final += 1
            final /= 2
            final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)

            final *=255

            name = f"{dataset}_image_{i}.jpg"

            cv2.imwrite(name, final)

parser = argparse.ArgumentParser(description="Dataset Image Exporter")
parser.add_argument('--dataset_directory', type=str, default='datasets', help='Location of the training data')
parser.add_argument('--num_images', default=10, type=int, help="Number of images per dataset.")
parser.add_argument('--resolution', default=512, type=int, help="Resolution of single images.")
args, _ = parser.parse_known_args()

dataset_directory = args.dataset_directory
num_images = args.num_images
resolution = args.resolution

datasets = get_subdir_names(dataset_directory)


for dataset in datasets:
    create_dataset_image(dataset, num_images, dataset_directory, resolution=resolution)
