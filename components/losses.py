import tensorflow as tf

from components.images import Images
from components.networks import Networks
from components.placeholders import Placeholders


class Losses:
    def __init__(self, networks: Networks, placeholders: Placeholders, images: Images, cycle_loss_coeff):
        self.discrimintate_images(networks, placeholders, images)
        self.define_losses(images, cycle_loss_coeff)

    def discrimintate_images(self, networks: Networks, placeholders: Placeholders, images: Images):
        self.D_real_a = networks.discriminator_a(images.image_a)
        self.D_fake_a = networks.discriminator_a(images.image_ba)
        self.D_real_b = networks.discriminator_b(images.image_b)
        self.D_fake_b = networks.discriminator_b(images.image_ab)
        self.D_history_fake_a = networks.discriminator_a(placeholders.history_fake_a_placeholder)
        self.D_history_fake_b = networks.discriminator_b(placeholders.history_fake_b_placeholder)

    def define_losses(self, images: Images, cycle_loss_coeff):
        # Least squre loss for GAN discriminator
        self.loss_D_a = (tf.reduce_mean(tf.squared_difference(self.D_real_a, 0.9)) +
                         tf.reduce_mean(tf.square(self.D_history_fake_a))) * 0.5
        self.loss_D_b = (tf.reduce_mean(tf.squared_difference(self.D_real_b, 0.9)) +
                         tf.reduce_mean(tf.square(self.D_history_fake_b))) * 0.5
        # Least squre loss for GAN generator
        self.loss_G_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_b, 0.9))
        self.loss_G_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_a, 0.9))
        # L1 norm for reconstruction error
        self.loss_rec_aba = tf.reduce_mean(tf.abs(images.image_a - images.image_aba))
        self.loss_rec_bab = tf.reduce_mean(tf.abs(images.image_b - images.image_bab))
        self.loss_cycle = cycle_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)
        self.loss_G_ab_final = self.loss_G_ab + self.loss_cycle
        self.loss_G_ba_final = self.loss_G_ba + self.loss_cycle