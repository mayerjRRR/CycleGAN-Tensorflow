import tensorflow as tf

from src.components.images import Images
from src.components.networks import Networks
from src.components.placeholders import Placeholders
from src.utils.tensor_ops import layer_frames_in_channels


class Losses:
    def __init__(self, networks: Networks, placeholders: Placeholders, images: Images, cycle_loss_coeff, train_videos,
                 train_images):
        self.define_discriminator_output(networks, placeholders, images)
        self.define_losses(images, placeholders, cycle_loss_coeff, train_videos, train_images)

    def define_discriminator_output(self, networks: Networks, placeholders: Placeholders, images: Images):
        self.define_discriminator_output_video(images, networks, placeholders)
        self.define_discriminator_output_images(images, networks, placeholders)
        self.define_temporal_discriminator_output(images, networks, placeholders)

    def define_discriminator_output_video(self, images: Images, networks: Networks, placeholders: Placeholders):
        self.D_real_frame_a = networks.discriminator_spatial_a(images.warped_frames_a[:, 1])
        self.D_fake_frame_a = networks.discriminator_spatial_a(images.warped_frames_ba[:, 1])
        self.D_real_frame_b = networks.discriminator_spatial_b(images.warped_frames_b[:, 1])
        self.D_fake_frame_b = networks.discriminator_spatial_b(images.warped_frames_ab[:, 1])
        self.D_history_fake_frame_a = networks.discriminator_spatial_a(
            placeholders.history_fake_temp_frames_a[:, 1])
        self.D_history_fake_frame_b = networks.discriminator_spatial_b(
            placeholders.history_fake_temp_frames_b[:, 1])

    def define_discriminator_output_images(self, images: Images, networks: Networks, placeholders: Placeholders):
        self.D_real_image_a = networks.discriminator_spatial_a(images.image_a)
        self.D_fake_image_a = networks.discriminator_spatial_a(images.image_ba)
        self.D_real_image_b = networks.discriminator_spatial_b(images.image_b)
        self.D_fake_image_b = networks.discriminator_spatial_b(images.image_ab)
        self.D_history_fake_image_a = networks.discriminator_spatial_a(placeholders.history_fake_a)
        self.D_history_fake_image_b = networks.discriminator_spatial_b(placeholders.history_fake_b)

    def define_temporal_discriminator_output(self, images: Images, networks: Networks, placeholders: Placeholders):

        self.D_temp_real_a = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_a))
        self.D_temp_fake_a = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_ba))
        self.D_temp_history_fake_a = networks.discriminator_temporal(
            layer_frames_in_channels(placeholders.history_fake_temp_frames_a))

        self.D_temp_real_b = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_b))
        self.D_temp_fake_b = networks.discriminator_temporal(layer_frames_in_channels(images.warped_frames_ab))
        self.D_temp_history_fake_b = networks.discriminator_temporal(
            layer_frames_in_channels(placeholders.history_fake_temp_frames_b))

    def define_losses(self, images: Images, placeholders:Placeholders, cycle_loss_coeff, train_videos, train_images):
        self.define_discriminator_loss(train_videos, train_images)
        self.define_spacial_generator_loss(train_images)
        self.define_temporal_generator_loss(placeholders, train_videos)
        self.define_cycle_loss(cycle_loss_coeff, images, train_images)
        self.define_total_generator_loss()

    def define_discriminator_loss(self, train_videos, train_images):
        if train_images:
            self.loss_D_a = (tf.reduce_mean(tf.squared_difference(self.D_real_image_a, 0.9)) +
                             tf.reduce_mean(tf.square(self.D_history_fake_image_a))) * 0.5
            self.loss_D_b = (tf.reduce_mean(tf.squared_difference(self.D_real_image_b, 0.9)) +
                             tf.reduce_mean(tf.square(self.D_history_fake_image_b))) * 0.5
        else:
            self.loss_D_a = (tf.reduce_mean(tf.squared_difference(self.D_real_frame_a, 0.9)) +
                             tf.reduce_mean(tf.square(self.D_history_fake_frame_a))) * 0.5
            self.loss_D_b = (tf.reduce_mean(tf.squared_difference(self.D_real_frame_b, 0.9)) + tf.reduce_mean(
                                tf.square(self.D_history_fake_frame_b))) * 0.5

        if train_videos:
            self.loss_D_temp = (tf.reduce_mean(tf.squared_difference(self.D_temp_real_a, 0.9)) +
                                (tf.reduce_mean(tf.square(self.D_temp_history_fake_a))) +
                                tf.reduce_mean(tf.squared_difference(self.D_temp_real_b, 0.9)) +
                                (tf.reduce_mean(tf.square(self.D_temp_history_fake_b)))) * 0.25
        else:
            self.loss_D_temp = tf.constant(0.0, dtype=tf.float32)

    def define_spacial_generator_loss(self, train_images):
        if train_images:
            self.loss_G_spat_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_image_b, 0.9))
            self.loss_G_spat_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_image_a, 0.9))
        else:
            self.loss_G_spat_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_frame_b, 0.9))
            self.loss_G_spat_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_frame_a, 0.9))

    def define_temporal_generator_loss(self, placeholders: Placeholders, train_videos):
        if train_videos:
            self.temp_loss_weigth = tf.clip_by_value((tf.cast(placeholders.global_step,dtype=tf.float32)-2000)/4000,0,1, name="temp_loss_weigth")
            self.loss_G_temp_ab = self.temp_loss_weigth*tf.reduce_mean(tf.squared_difference(self.D_temp_fake_b, 0.9))
            self.loss_G_temp_ba = self.temp_loss_weigth*tf.reduce_mean(tf.squared_difference(self.D_temp_fake_a, 0.9))
        else:
            self.loss_G_temp_ab = self.loss_G_temp_ba = tf.constant(0.0, dtype=tf.float32)

    def define_cycle_loss(self, cycle_loss_coeff, images: Images, train_images):
        if train_images:
            self.loss_rec_aba = tf.reduce_mean(tf.abs(images.image_a - images.image_aba))
            self.loss_rec_bab = tf.reduce_mean(tf.abs(images.image_b - images.image_bab))
        else:
            self.loss_rec_aba = tf.reduce_mean(tf.abs(images.warped_frames_a[:, 1] - images.frames_aba[:, 1]))
            self.loss_rec_bab = tf.reduce_mean(tf.abs(images.warped_frames_b[:, 1] - images.frames_bab[:, 1]))

        self.loss_cycle = cycle_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)

    def define_total_generator_loss(self):
        self.loss_G_ab_final = self.loss_G_spat_ab + self.loss_G_temp_ab + self.loss_cycle
        self.loss_G_ba_final = self.loss_G_spat_ba + self.loss_G_temp_ba + self.loss_cycle
