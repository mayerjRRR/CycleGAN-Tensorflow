from src.efficient_data_loader import get_training_datasets
import tensorflow as tf
import cv2



train_A, train_B, test_A, test_B = get_training_datasets('videos', 256, 4)

next_a = test_A.make_one_shot_iterator().get_next()
next_a = test_A.make_one_shot_iterator().get_next()
next_b = train_B.make_one_shot_iterator().get_next()

with tf.Session() as sess:

    while True:
        b = sess.run(next_a)[0]
        i = 0
        for frame in b:
            cv2.imshow(f"frame #{i}", frame)
            i+=1
        cv2.waitKey(1000)
