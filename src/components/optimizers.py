import tensorflow as tf

from src.components.losses import Losses
from src.components.networks import Networks
from src.components.placeholders import Placeholders


class Optimizers:
    def __init__(self, networks: Networks, losses: Losses, placeholders: Placeholders, train_videos):
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_D, var_list=networks.discriminator.var_list, global_step=placeholders.global_step)
            # TODO: Train with L1/L2-Loss eventually
            #fnet_variable_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
            #self.optimizer_fnet = tf.train.AdamOptimizer(learning_rate=placeholders.lr*0.5, beta1=0.5) \
            #    .minimize(losses.loss_D_temp, var_list=fnet_variable_list)

            # self.optimizer_fnet = tf.constant(0,dtype=tf.float32)

        self.optimizer_G_ab = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_G_ab_final, var_list=networks.generator_ab.var_list)
        self.optimizer_G_ba = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_G_ba_final, var_list=networks.generator_ba.var_list)