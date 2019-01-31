"""
fnet from TecoGAN
simply try with:
python3 fnet.py ( Rachel: only tested with python3 )
"""

import tensorflow as tf, numpy as np, scipy.misc, os
import tensorflow.contrib.slim as slim
import keras

############################################ basic tensorflow functions #######################################

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)


# Define our Lrelu
def lrelu(inputs, alpha):
    return keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def maxpool(inputs, scope='maxpool'):
    return slim.max_pool2d(inputs, [2, 2], scope=scope)


def print_variables(scope, sess, key=tf.GraphKeys.MODEL_VARIABLES):
    print("Scope %s:" % scope)
    variables_names = [v.name for v in tf.get_collection(key, scope=scope)]
    # variables_names = sorted(variables_names)
    values = sess.run(variables_names)
    total_sz = 0
    for k, v in zip(variables_names, values):
        print("Variable: " + k + ", shape:" + str(v.shape))
        total_sz += np.prod(v.shape)
    print("total size: %d" % total_sz)


############################################ fnet #############################################################
# Definition of the fnet, more details can be found in FRVSR paper
def fnet(fnet_input, reuse=False):
# fnet_input, batch*(frame-1), h, w, sn*2
    
    def down_block( inputs, output_channel = 64, stride = 1, scope = 'down_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = lrelu(net, 0.2)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = lrelu(net, 0.2)
            net = maxpool(net)
        return net
        
    def up_block( inputs, output_channel = 64, stride = 1, scope = 'up_block'):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
            net = lrelu(net, 0.2)
            net = conv2(net, 3, output_channel, stride, use_bias=True, scope='conv_2')
            net = lrelu(net, 0.2)
            new_shape = tf.shape(net)[1:-1]*2
            net = tf.image.resize_images(net, new_shape)
        return net
        
    with tf.variable_scope('autoencode_unit', reuse=reuse):
        net = down_block( fnet_input, 32, scope = 'encoder_1')
        net = down_block( net, 64, scope = 'encoder_2')
        net = down_block( net, 128, scope = 'encoder_3')
        
        net = up_block( net, 256, scope = 'decoder_1')
        net = up_block( net, 128, scope = 'decoder_2')
        net1 = up_block( net, 64, scope = 'decoder_3')
        
        with tf.variable_scope('output_stage'):
            net = conv2(net1, 3, 32, 1, scope='conv1')
            net = lrelu(net, 0.2)
            net2 = conv2(net, 3, 2, 1, scope='conv2')
            net = tf.tanh(net2) * 24.0 # a parameter for max velocity
            
    return net # batch*(frame-1), FLAGS.crop_size, FLAGS.crop_size, 2


############################################ IO functions #####################################################
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


def save_map(out_path, estV, estE):
    """
        estV, shape (h,w,2), estE, shape (h,w,1)
    """
    drawV = np.abs(estV) 
    img = np.concatenate( [drawV/24.0, estE/estE.max()], axis=-1 )
    # R: vel_y, G: vel_x, B: match_error
    save_img(out_path, img)


############################################ a simple test ####################################################
def simple_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # load the test data
    prev_input = get_img('./fnet/col_high_0000.png')
    current_input = get_img('./fnet/col_high_0001.png')
    # concatenated together as input for flow estimation
    input_fr = np.concatenate( [[prev_input], [current_input]], axis=-1 )
    print(input_fr.shape)
    # build the fnet
    inputs_frames = tf.placeholder(tf.float32, shape=input_fr.shape, name='inputs_frames')
    with tf.variable_scope('fnet'):
        est_flow = fnet( inputs_frames, reuse=False )
        # est_flow.shape is 8 * (input_fr.shape // 8), so it may be a little smaller (0-7 pixels) than input_fr
        est_flow = tf.image.resize_images(est_flow, input_fr.shape[1:-1])
    
    # apply the flow to warp the previous image, and check the error
    pre_input_warp = tf.contrib.image.dense_image_warp( inputs_frames[:,:,:,:3], est_flow)
    warp_diff_abs = tf.keras.backend.abs(inputs_frames[:,:,:,3:] - pre_input_warp)
    warp_diff_l1 = tf.reduce_sum(warp_diff_abs, axis=[3], keep_dims=True)
    
    # create saver to load pre-trained model
    var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
    weight_initiallizer = tf.train.Saver(var_list)
    
    # Define the initialization operation
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(local_init_op)
        print_variables('fnet', sess)
        print('Loading weights from ckpt model ./fnet/fnet-0')
        weight_initiallizer.restore(sess, './fnet/fnet-0')
        print('run simple test with col_high_0000 - 0001 images')
        preV, warpPre, warpE = sess.run((est_flow, pre_input_warp, warp_diff_l1), feed_dict={inputs_frames: input_fr})
        save_img('warp0to1.png', warpPre[0])
        save_map('map0to1.png', preV[0], warpE[0])
        print('Done. Please check warp0to1.png and map0to1.png')


if __name__ == '__main__':
    simple_test()
    
    
######################################## Intro to Tempo Discriminator #########################################
"""
    Tempo discriminator is similar to the normal (spatial) discriminator, only has a different input.
    Its input is the concatenation of the warped previous frame, the current frame and the next frame.
    For example, if you have tensors like x_pre, x_cur and x_nxt in shape of (batch, height, width, channel).
    pre2cur_Vel = fnet( tf.concat( (x_pre, x_cur), axis=-1 ), reuse=False)
    nxt2cur_Vel = fnet( tf.concat( (x_nxt, x_cur), axis=-1 ), reuse=True)
    
    # 1. remember to resize the velocity properly
    # 2. limitation for dense_image_warp, shapes of inputs should be known. None or -1 are not allowed.
    #    Shapes can be fixed with :  pre2cur_Vel.set_shape( (16, 256, 256, 2) )
    warped_pre = tf.contrib.image.dense_image_warp( x_pre, pre2cur_Vel)
    warped_nxt = tf.contrib.image.dense_image_warp( x_nxt, nxt2cur_Vel)
    tempo_input = tf.concat( (warped_pre, x_cur, warped_nxt), axis=-1 )
    # 3. In fnet, convolutional filters do not have enough neighbor pixels (<12 pixels) for border regions,
    #    therefore borders won't have best quality, and this may mislead the discriminator. Try:
    offset_dt = 6
    tempo_input = tf.image.crop_to_bounding_box(tempo_input, offset_dt, offset_dt, h-2*offset_dt, w-2*offset_dt)
    
    # can be used for tempoDiscriminator then:
    discrim_output = tempoDiscriminator( tempo_input_generated or tempo_input_target )
    
"""