import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
import numpy as np
import scipy
from src.components.savers import Saver


def save_img(out_path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')  # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)

    return np.array(img, dtype=np.float32) / 255.0



inputs = tf.placeholder(tf.float32,shape=[4,256,256,3])

with slim.arg_scope(vgg.vgg_arg_scope()):
    _, end_points = vgg.vgg_19(inputs, spatial_squeeze=False)
    print("pen")
    print(end_points)

print("atu")
fnet_variable_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="vgg_19")
saver = Saver(fnet_variable_list,save_path="vgg19")
print("asffas")
with tf.Session() as sess:
    saver.load(sess)
    print(end_points['vgg_19/conv4/conv4_2'])
