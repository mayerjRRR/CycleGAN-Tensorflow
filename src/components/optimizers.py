import tensorflow as tf

from src.components.losses import Losses
from src.components.networks import Networks
from src.components.placeholders import Placeholders


class Optimizers:
    def __init__(self, networks: Networks, losses: Losses, placeholders: Placeholders, train_videos, tb_threshold):

        adam = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5)
        def optimize_conditionally(loss, var_list, global_step = None):
            with tf.control_dependencies([adam.minimize(loss, var_list=var_list, global_step=global_step)]):
                return tf.constant(0,dtype=tf.float32)


        if (train_videos):
            self.optimizer_D_temp = tf.cond(tf.less(placeholders.tb_temporal, tb_threshold), lambda: optimize_conditionally(losses.loss_D_temp, var_list=networks.discriminator_temporal.var_list), lambda: tf.constant(0, dtype=tf.float32))
            self.optimizer_D_a = tf.cond(tf.less(placeholders.tb_spatial_a, tb_threshold),
                                         lambda: optimize_conditionally(losses.loss_D_a,
                                                                        var_list=networks.discriminator_spatial_a.var_list,
                                                                        global_step=placeholders.global_step),
                                         lambda: tf.constant(0, dtype=tf.float32))
            self.optimizer_D_b = tf.cond(tf.less(placeholders.tb_spatial_b, tb_threshold),
                                         lambda: optimize_conditionally(losses.loss_D_b,
                                                                        var_list=networks.discriminator_spatial_b.var_list),
                                         lambda: tf.constant(0, dtype=tf.float32))
        else:
            self.optimizer_D_temp = tf.constant(0, dtype=tf.float32)
            self.optimizer_D_a = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_D_a,var_list=networks.discriminator_spatial_a.var_list)
            self.optimizer_D_b = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_D_b, var_list=networks.discriminator_spatial_b.var_list)


        self.optimizer_G_ab = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_G_ab_final, var_list=networks.generator_ab.var_list,global_step=placeholders.global_step)
        self.optimizer_G_ba = tf.train.AdamOptimizer(learning_rate=placeholders.lr, beta1=0.5) \
            .minimize(losses.loss_G_ba_final, var_list=networks.generator_ba.var_list)