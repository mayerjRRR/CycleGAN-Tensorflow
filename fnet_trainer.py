from src.data_loader import get_training_datasets
from fnet.fnet import fnet
import tensorflow as tf
from time import gmtime, strftime

def train(image_size = 256, batch_size = 8, dataset_directory="datasets", task="vidzebra"):


    train_A, train_B = get_training_datasets(task, image_size, batch_size,
                                             dataset_dir=dataset_directory, frame_sequence_length=2)

    next_a = train_B.make_one_shot_iterator().get_next()
    reshaped = tf.transpose(next_a, perm=[0, 2, 3, 1, 4])
    transposed = tf.reshape(reshaped, [batch_size, image_size, image_size, 6]);

    with tf.variable_scope('fnet'):
        est_flow = fnet(transposed, reuse=False)
        # est_flow.shape is 8 * (input_fr.shape // 8), so it may be a little smaller (0-7 pixels) than input_fr
        #est_flow = tf.image.resize_images(est_flow, input_fr.shape[1:-1])

    # apply the flow to warp the previous image, and check the error
    first, second = tf.split(transposed, num_or_size_splits=2, axis=-1)

    pre_input_warp = tf.contrib.image.dense_image_warp(first, est_flow)
    warp_diff_abs = tf.abs(second - pre_input_warp)
    warp_diff_l1 = tf.reduce_sum(warp_diff_abs, axis=[3], keep_dims=True)

    loss = tf.reduce_mean(warp_diff_abs)

    # create saver to load pre-trained model
    var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
    saver = tf.train.Saver(var_list)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)


    tf.summary.scalar('L1-loss', loss)
    tf.summary.image('rrr/First', first[0:1])
    tf.summary.image('rrr/Second', second[0:1])
    tf.summary.image('rrr/Warped', pre_input_warp[0:1])
    tf.summary.image('rrr/Error', warp_diff_l1[0:1])
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./fnet/log/'+strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

    # Define the initialization operation
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(local_init_op)
        saver.restore(sess, './fnet/fnet-0')
        train_writer.add_graph(sess.graph)

        iteration = 0;
        while True:
            to_run = [optimizer, loss]
            if iteration % 100 == 0:
                to_run += [summary_op]
            result = sess.run(to_run)
            if iteration % 100 == 0:
                train_writer.add_summary(result[-1], iteration)
                train_writer.flush()
                print("writing log")
            if iteration % 1000 == 0:
                print("saving...")
                saver.save(sess, './fnet/trained/fnet', global_step=iteration)
            print ("Loss: "+str(result[1]),end="\r")
            iteration+=1



if __name__ == '__main__':
    train()