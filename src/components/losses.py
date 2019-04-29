import tensorflow as tf

from src.components.images import Images
from src.components.networks import Networks
from src.components.placeholders import Placeholders
from src.utils.tensor_ops import generate_temp_discriminator_input
from src.utils.argument_parser import TrainingConfig
from src.utils.warp_utils import compute_pingpong_difference


class Losses:
    def __init__(self, networks: Networks, placeholders: Placeholders, images: Images, training_config: TrainingConfig,
                 train_videos, train_images):
        self.define_discriminator_output(networks, placeholders, images)
        self.define_losses(images, placeholders, training_config, train_videos, train_images)

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

        self.D_temp_real_a = networks.discriminator_temporal(generate_temp_discriminator_input(images.warped_frames_a))
        self.D_temp_fake_a = networks.discriminator_temporal(generate_temp_discriminator_input(images.warped_frames_ba))
        self.D_temp_history_fake_a = networks.discriminator_temporal(
            generate_temp_discriminator_input(placeholders.history_fake_temp_frames_a))

        self.D_temp_real_b = networks.discriminator_temporal(generate_temp_discriminator_input(images.warped_frames_b))
        self.D_temp_fake_b = networks.discriminator_temporal(generate_temp_discriminator_input(images.warped_frames_ab))
        self.D_temp_history_fake_b = networks.discriminator_temporal(
            generate_temp_discriminator_input(placeholders.history_fake_temp_frames_b))

    def define_losses(self, images: Images, placeholders: Placeholders, training_config, train_videos, train_images):
        self.define_discriminator_loss(train_videos, train_images)
        self.define_spatial_generator_loss(train_images)
        self.define_temporal_generator_loss(placeholders, train_videos, training_config.temporal_loss_coefficient)
        self.define_cycle_loss(images, training_config.cycle_loss_coefficient, train_images)
        self.define_identity_loss(images, placeholders, training_config.identity_loss_coefficient, train_images)
        self.define_code_layer_loss(images, training_config.code_loss_coefficient, train_images)
        self.define_pingpong_loss(images, training_config.pingpong_loss_coefficient, train_videos)
        self.define_total_generator_loss()

    def define_discriminator_loss(self, train_videos, train_images):
        with tf.name_scope("discriminator_loss"):
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

    def define_spatial_generator_loss(self, train_images):
        with tf.name_scope("spatial_loss"):
            if train_images:
                self.loss_G_spat_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_image_b, 0.9))
                self.loss_G_spat_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_image_a, 0.9))
            else:
                self.loss_G_spat_ab = tf.reduce_mean(tf.squared_difference(self.D_fake_frame_b, 0.9))
                self.loss_G_spat_ba = tf.reduce_mean(tf.squared_difference(self.D_fake_frame_a, 0.9))

    def define_temporal_generator_loss(self, placeholders: Placeholders, train_videos, temp_loss_coeff):
        with tf.name_scope("temporal_loss"):
            self.temp_loss_fade_in_weigth = fade_in_weight(placeholders.global_step, 2000, 4000, "temp_loss_weigth")
            if train_videos:
                self.loss_G_temp_ab =  temp_loss_coeff * tf.reduce_mean(tf.squared_difference(self.D_temp_fake_b, 0.9))
                self.loss_G_temp_ba =  temp_loss_coeff * tf.reduce_mean(tf.squared_difference(self.D_temp_fake_a, 0.9))
            else:
                self.loss_G_temp_ab = self.loss_G_temp_ba = tf.constant(0.0, dtype=tf.float32)

    def define_cycle_loss(self, images: Images, cycle_loss_coeff, train_images):
        with tf.name_scope("cycle_loss"):
            if train_images:
                self.loss_rec_aba = tf.reduce_mean(tf.abs(images.image_a - images.image_aba))
                self.loss_rec_bab = tf.reduce_mean(tf.abs(images.image_b - images.image_bab))
            else:
                self.loss_rec_aba = tf.reduce_mean(tf.abs(images.frames_a - images.frames_aba))
                self.loss_rec_bab = tf.reduce_mean(tf.abs(images.frames_b - images.frames_bab))

            self.loss_cycle = cycle_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)

    def define_identity_loss(self, images: Images, placeholders: Placeholders, identity_loss_coeff, train_images):
        with tf.name_scope("identity_loss"):
            if train_images:
                self.loss_id_ab = tf.reduce_mean(tf.abs(images.image_a - images.image_ab))
                self.loss_id_ba = tf.reduce_mean(tf.abs(images.image_b - images.image_ba))
            else:
                self.loss_id_ab = tf.reduce_mean(tf.abs(images.frames_a - images.frames_ab))
                self.loss_id_ba = tf.reduce_mean(tf.abs(images.frames_b - images.frames_ba))

            self.identity_fade_out_weight = fade_out_weight(placeholders.global_step, 1000, 1500, "identity_fade_out_weight")
            self.loss_identity = identity_loss_coeff * (self.loss_rec_aba + self.loss_rec_bab)

    def define_pingpong_loss(self, images: Images, pingpong_loss_coeff, train_videos):
        with tf.name_scope("ping_pong_loss"):
            if train_videos:
                self.loss_pingpong_ab = pingpong_loss_coeff*tf.reduce_mean(compute_pingpong_difference(images.pingpong_frames_ab))
                self.loss_pingpong_ba = pingpong_loss_coeff*tf.reduce_mean(compute_pingpong_difference(images.pingpong_frames_ba))
            else:
                self.loss_pingpong_ab = self.loss_pingpong_ba = tf.constant(0.0, dtype=tf.float32)

    def define_code_layer_loss(self, images: Images, code_loss_coeff, train_images):
        with tf.name_scope("code_layer_loss"):
            if train_images:
                self.loss_code_aba = self.loss_code_bab = tf.constant(0.0, dtype=tf.float32)
            else:
                self.loss_code_aba = tf.reduce_mean(tf.abs(images.code_frames_ab - images.code_frames_aba))
                self.loss_code_bab = tf.reduce_mean(tf.abs(images.code_frames_ba - images.code_frames_bab))
            self.loss_code = code_loss_coeff*(self.loss_code_aba + self.loss_code_bab)

    def define_total_generator_loss(self):
        with tf.name_scope("generator_loss"):
            self.loss_G_ab_final = self.loss_G_spat_ab + self.temp_loss_fade_in_weigth * (self.loss_G_temp_ab + self.loss_pingpong_ab) + self.loss_cycle + self.identity_fade_out_weight * self.loss_identity + self.loss_code
            self.loss_G_ba_final = self.loss_G_spat_ba + self.temp_loss_fade_in_weigth * (self.loss_G_temp_ba + self.loss_pingpong_ba) + self.loss_cycle + self.identity_fade_out_weight * self.loss_identity + self.loss_code


def fade_in_weight(step, start, duration, name):
    return tf.clip_by_value((tf.cast(step, dtype=tf.float32) - start) / duration, 0, 1, name=name)


def fade_out_weight(step, start, duration, name):
    return 1 - fade_in_weight(step, start, duration, name)
