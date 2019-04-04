from src.data_loader import get_training_datasets
import tensorflow as tf
import src.utils.tensor_ops as to
import numpy as np
import cv2

a, _ = get_training_datasets("vidzebra",256,4,"datasets",3)

raw = tf.reshape(a.make_one_shot_iterator().get_next(),[4,3,256,256,3])
print(raw.get_shape().as_list())
stacked = to.layer_frames_in_channels(raw)
print(stacked.get_shape().as_list())
recovered = to.extract_frames_from_channels(stacked)
print(recovered.get_shape().as_list())

sess = tf.Session()
raw_im, stacked_im, recovered_im = sess.run([raw,stacked,recovered])

print(raw_im.shape)

cv2.imshow("raw",raw_im[0,0])
cv2.imshow("recoverd",recovered_im[0,0])

diff = raw_im-recovered_im

print("error "+str(np.sum(diff)))

cv2.waitKey(0)