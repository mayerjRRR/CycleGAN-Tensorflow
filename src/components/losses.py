import tensorflow as tf

from src.components.images import Images
from src.components.networks import Networks
from src.components.placeholders import Placeholders
from src.utils.tensor_ops import layer_frames_in_channels
from src.utils.argument_parser import TrainingConfig
from src.utils.warp_utils import compute_pingpong_difference


class Losses:
    def __init__(self, networks: Networks, placeholders: Placeholders, images: Images, training_config: TrainingConfig,
                 train_videos, train_images):
        self.define_discriminator_output(networks, placeholders, images)
        self.define_losses(images, placeholders, training_config, train_videos, train_images)

    def define_discriminator_output(self, networks: Networks, placeholders: Placeholders, images: Images):
        self.define_discriminator_output_video(images, networks, placeholders)

    def define_discriminator_output_video(self, images: Images, networks: Networks, placeholders: Placeholders):
        self.D_real_frame_a = networks.discriminator_spatial_a(layer_frames_in_channels(images.warped_frames_a))
        self.D_fake_frame_a = networks.discriminator_spatial_a(layer_frames_in_channels(images.warped_frames_ba))
        self.D_real_frame_b = networks.discriminator_spatial_b(layer_frames_in_channels(images.warped_frames_b))
        self.D_fake_frame_b = networks.discriminator_spatial_b(layer_frames_in_channels(images.warped_frames_ab))
        self.D_history_fake_frame_a = networks.discriminator_spatial_a(
            layer_frames_in_channels(placeholders.history_fake_temp_frames_a))
        self.D_history_fake_frame_b = networks.discriminator_spatial_b(
            layer_frames_in_channels(placeholders.history_fake_temp_frames_b))

    def define_discriminator_output_images(self, images: Images, networks: Networks, placeholders: Placeholders):
        self.D_real_image_a = networks.discriminator_spatial_a(images.image_a)
        self.D_fake_image_a = networks.discriminator_spatial_a(images.image_ba)
        self.D_real_image_b = networks.discriminator_spatial_b(images.image_b)
        self.D_fake_image_b = networks.discriminator_spatial_b(images.image_ab)
        self.D_history_fake_image_a = networks.discriminator_spatial_a(placeholders.history_fake_a)
        self.D_history_fake_image_b = networks.discriminator_spatial_b(placeholders.history_fake_b)

    def define_losses(self, images: Images, placeholders: Placeholders, training_config, train_videos, train_images):
        self.define_discriminator_loss(train_videos, train_images)
        self.define_spacial_generator_loss(train_images)
        self.define_cycle_loss(images, training_config.cycle_loss_coefficient, train_images)
        self.define_identity_loss(images, placeholders, training_config.identity_loss_coefficient, train_images)
        self.define_pingpong_loss(images, training_config.pingpong_loss_coefficient, train_videos, placeholders)
        self.define_total_generator_loss()

    def define_discriminator_loss(self, train_videos, train_images):
        self.loss_D_a = (tf.reduce_mean(tf.squared_difference(self.D_real_frame_a, 0.9)) +
                         tf.reduce_mean(tf.square(self.D_history_fake_frame_a))) * 0.5
        self.loss_D_b = (tf.reduce_mean(tf.squared_difference(self.D_real_frame_b, 0.9)) + tf.reduce_mean(
            tf.square(self.D_history_fake_frame_b))) * 0.5

    def define_spacial_generator_loss(self, train_images):
        self.loss_G_spat_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_frame_b, 0.9))
        self.loss_G_spat_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_frame_a, 0.9))

    def define_cycle_loss(self, images: Images, cycle_loss_coeff, train_images):

        self.loss_rec_aba = tf.reduce_mean(tf.abs(images.warped_frames_a[:, 1] - images.frames_aba[:, 1]))
        self.loss_rec_bab = tf.reduce_mean(tf.abs(images.warped_frames_b[:, 1] - images.frames_bab[:, 1]))

        self.loss_cycle = cycle_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)

    def define_identity_loss(self, images: Images, placeholders: Placeholders, identity_loss_coeff, train_images):

        self.loss_id_ab = tf.reduce_mean(tf.abs(images.warped_frames_a[:, 1] - images.warped_frames_ab[:, 1]))
        self.loss_id_ba = tf.reduce_mean(tf.abs(images.warped_frames_b[:, 1] - images.warped_frames_ba[:, 1]))

        self.identity_fade_out_weight = fade_out_weight(placeholders.global_step, 1000, 1500,
                                                        "identity_fade_out_weight")
        self.loss_identity = identity_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)

    def define_pingpong_loss(self, images: Images, pingpong_loss_coeff, train_videos, placeholders):

        self.temp_loss_fade_in_weigth = fade_in_weight(placeholders.global_step, 2000, 4000, "temp_loss_weigth")
        self.loss_pingpong_ab = pingpong_loss_coeff * tf.reduce_mean(
            compute_pingpong_difference(images.pingpong_frames_ab))
        self.loss_pingpong_ba = pingpong_loss_coeff * tf.reduce_mean(
            compute_pingpong_difference(images.pingpong_frames_ba))


    def define_total_generator_loss(self):
        self.loss_G_ab_final = self.loss_G_spat_ab + self.temp_loss_fade_in_weigth * (
            self.loss_pingpong_ab) + self.loss_cycle + self.identity_fade_out_weight * self.loss_identity
        self.loss_G_ba_final = self.loss_G_spat_ba + self.temp_loss_fade_in_weigth * (
            self.loss_pingpong_ba) + self.loss_cycle + self.identity_fade_out_weight * self.loss_identity


def fade_in_weight(step, start, duration, name):
    return tf.clip_by_value((tf.cast(step, dtype=tf.float32) - start) / duration, 0, 1, name=name)


def fade_out_weight(step, start, duration, name):
    return 1 - fade_in_weight(step, start, duration, name)
