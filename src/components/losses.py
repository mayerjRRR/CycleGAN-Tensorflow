import tensorflow as tf

from src.components.images import Images
from src.components.networks import Networks
from src.components.placeholders import Placeholders


class Losses:
    def __init__(self, networks: Networks, placeholders: Placeholders, images: Images, cycle_loss_coeff):
        self.define_discriminator_output(networks, placeholders, images)
        self.define_losses(images, cycle_loss_coeff)

    def define_discriminator_output(self, networks: Networks, placeholders: Placeholders, images: Images):
        self.D_real_a = networks.discriminator_spatial_a(images.warped_frames_a[:, 1])
        self.D_fake_a = networks.discriminator_spatial_a(images.frames_ba[:, 1])
        self.D_real_b = networks.discriminator_spatial_b(images.warped_frames_b[:, 1])
        self.D_fake_b = networks.discriminator_spatial_b(images.frames_ab[:, 1])
        self.D_history_fake_a = networks.discriminator_spatial_a(placeholders.history_fake_warped_frames_a_placeholder[:, 1])
        self.D_history_fake_b = networks.discriminator_spatial_b(placeholders.history_fake_warped_frames_b_placeholder[:, 1])

        #TODO: Move to temp discriminator
        def layer_frames_in_channels(input_frames):
            reshaped = tf.transpose(input_frames, perm=[0, 2, 3, 1, 4])
            shape = reshaped.get_shape().as_list()
            shape[-2] *= shape[-1]
            shape.pop();
            return tf.reshape(reshaped, shape);

        self.D_temp_real_a = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_a))
        self.D_temp_fake_a = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_ba))
        self.D_temp_history_fake_a = networks.discriminator_temporal(layer_frames_in_channels(placeholders.history_fake_warped_frames_a_placeholder))

        self.D_temp_real_b = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_b))
        self.D_temp_fake_b = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_ab))
        self.D_temp_history_fake_b = networks.discriminator_temporal(layer_frames_in_channels(placeholders.history_fake_warped_frames_b_placeholder))

    def define_losses(self, images: Images, cycle_loss_coeff):
        self.define_discriminator_loss()
        self.define_spacial_generator_loss()
        self.define_temporal_generator_loss()
        self.define_cycle_loss(cycle_loss_coeff, images)
        self.define_total_generator_loss()

    def define_discriminator_loss(self):
        self.loss_D_a = (tf.reduce_mean(tf.squared_difference(self.D_real_a, 0.9)) +
                         tf.reduce_mean(tf.square(self.D_history_fake_a))) * 0.5
        self.loss_D_b = (tf.reduce_mean(tf.squared_difference(self.D_real_b, 0.9)) +
                         tf.reduce_mean(tf.square(self.D_history_fake_b))) * 0.5

        self.loss_D_temp = (tf.reduce_mean(tf.squared_difference(self.D_temp_real_a, 0.9)) +
                            (tf.reduce_mean(tf.square(self.D_temp_history_fake_a))) +
                            tf.reduce_mean(tf.squared_difference(self.D_temp_real_b, 0.9)) +
                            (tf.reduce_mean(tf.square(self.D_temp_history_fake_b)))) * 0.25

    def define_spacial_generator_loss(self):
        self.loss_G_spat_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_b, 0.9))
        self.loss_G_spat_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_a, 0.9))

    def define_temporal_generator_loss(self):
        self.loss_G_temp_ab = tf.reduce_mean(tf.squared_difference(self.D_temp_fake_b, 0.9))
        self.loss_G_temp_ba = tf.reduce_mean(tf.squared_difference(self.D_temp_fake_a, 0.9))

    def define_cycle_loss(self, cycle_loss_coeff, images):
        self.loss_rec_aba = tf.reduce_mean(tf.abs(images.warped_frames_a[:, 1] - images.frames_aba[:, 1]))
        self.loss_rec_bab = tf.reduce_mean(tf.abs(images.warped_frames_b[:, 1] - images.frames_bab[:, 1]))
        self.loss_cycle = cycle_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)

    def define_total_generator_loss(self):
        self.loss_G_ab_final = self.loss_G_spat_ab + 0.2*self.loss_G_temp_ab + self.loss_cycle
        self.loss_G_ba_final = self.loss_G_spat_ba + 0.2*self.loss_G_temp_ba + self.loss_cycle
