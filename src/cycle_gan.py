import numpy as np
from tqdm import trange

from src.components.images import Images
from src.components.losses import Losses
from src.components.networks import Networks
from src.components.optimizers import Optimizers
from src.components.placeholders import Placeholders
from src.components.savers import Savers
from src.components.tensor_board_summary import TensorBoardSummary
from src.utils.history_queue import HistoryQueue
from src.utils.utils import logger


class CycleGan(object):

    def __init__(self, save_dir, image_height=256, image_width=None, batch_size=4, cycle_loss_coeff=1, log_step=10,
                 train_videos=True, train_images=False):
        self.init_parameters(image_height, image_width, batch_size, cycle_loss_coeff, log_step, train_videos,
                             train_images)

        self.placeholders = Placeholders(self._batch_size, self._image_shape)
        self.networks = Networks(self.placeholders)

        self.images = Images(self.placeholders, self.networks, self._image_shape, self._batch_size, self._augment_shape)
        self.losses = Losses(self.networks, self.placeholders, self.images, self._cycle_loss_coeff, self.train_videos,
                             self.train_images)
        self.optimizers = Optimizers(self.networks, self.losses, self.placeholders, self.train_videos)
        self.tb_summary = TensorBoardSummary(self.images, self.losses, self.placeholders, self.train_videos,
                                             self.train_images)

        self.savers = Savers(self.networks, self.placeholders, save_dir)

    def init_parameters(self, image_height, image_width, batch_size, cycle_loss_coeff, log_step, train_videos,
                        train_images):
        self.init_args(image_height, image_width, batch_size, cycle_loss_coeff, log_step, 550, train_videos,
                       train_images)
        self.init_image_dimensions()

    def init_args(self, image_height, image_width, batch_size, cycle_loss_coeff, log_step, save_step, train_videos,
                  train_images):
        self._log_step = log_step
        self._save_step = save_step
        self._batch_size = batch_size
        self._image_height = image_height
        self._cycle_loss_coeff = cycle_loss_coeff
        self._save_step = save_step
        if image_width is None:
            image_width = image_height
        self._image_width = image_width
        self.train_videos = train_videos
        self.train_images = train_images

    def init_image_dimensions(self):
        self._augment_shape = [self._image_height + int(self._image_height / 8),
                               self._image_width + int(self._image_width / 8)]
        self._image_shape = [self._image_height, self._image_width, 3]

    def train_on_videos(self, sess, summary_writer, frame_data_a, frame_data_b):
        epoch_length, history_a, history_b, lr_decay, lr_initial, num_initial_iter, steps = self.init_training(sess)
        for step in steps:
            lr = self.get_learning_rate(step, epoch_length, lr_decay, lr_initial, num_initial_iter)

            frames_a, frames_b = self.get_real_frames(frame_data_a, frame_data_b, sess)
            # fakes_a, fakes_b = self.get_fake_frames(frames_a, frames_b, sess)
            # warped_frames_a, warped_frames_b, warped_fakes_a, warped_fakes_b = self.get_warped_images(frames_a,
            #                                                                                           frames_b,
            #                                                                                           fakes_a, fakes_b,
            #                                                                                           sess)
            warped_fakes_a, warped_fakes_b = sess.run([self.images.warped_frames_ba, self.images.warped_frames_ab],
                                                      feed_dict={self.placeholders.frames_a: frames_a,
                                                                 self.placeholders.frames_b: frames_b,
                                                                 self.placeholders.is_train: True,
                                                                 self.placeholders.lr: lr})

            fake_a_history, fake_b_history = self.query_history_queue(warped_fakes_a, warped_fakes_b, history_a,
                                                                      history_b)

            fetches = self.get_fetches(step, video_training=True)

            fetched = sess.run(fetches, feed_dict={self.placeholders.frames_a: frames_a,
                                                   self.placeholders.frames_b: frames_b,
                                                   self.placeholders.is_train: True,
                                                   self.placeholders.lr: lr,
                                                   self.placeholders.history_fake_warped_frames_a: fake_a_history,
                                                   self.placeholders.history_fake_warped_frames_b: fake_b_history})
            if self.should_write_summary(step):
                self.write_summary(fetched, step, steps, summary_writer)
            if self.should_save_model(step):
                self.savers.save_all(sess, global_step=step)

    def train_on_images(self, sess, summary_writer, image_data_a, image_data_b):
        # TODO: dont't fetch temp_disc optimizer
        epoch_length, history_a, history_b, lr_decay, lr_initial, num_initial_iter, steps = self.init_training(sess)
        for step in steps:
            lr = self.get_learning_rate(step, epoch_length, lr_decay, lr_initial, num_initial_iter)

            image_a, image_b = self.get_real_images(image_data_a, image_data_b, sess)
            fake_a, fake_b = self.get_fake_images(image_a, image_b, sess)

            fake_a_history, fake_b_history = self.query_history_queue(fake_a, fake_b, history_a,
                                                                      history_b)

            fetches = self.get_fetches(step, video_training=False)

            fetched = sess.run(fetches, feed_dict={self.placeholders.image_a: image_a,
                                                   self.placeholders.image_b: image_b,
                                                   self.placeholders.is_train: True,
                                                   self.placeholders.lr: lr,
                                                   self.placeholders.history_fake_a: fake_a_history,
                                                   self.placeholders.history_fake_b: fake_b_history})
            if self.should_write_summary(step):
                self.write_summary(fetched, step, steps, summary_writer)
            if self.should_save_model(step):
                self.savers.save_all(sess)

    def init_training(self, sess):
        logger.info('Start training.')
        epoch_length, initial_step, lr_decay, lr_initial, num_global_step, num_initial_iter = \
            self.init_training_parameters(sess)
        history_a = HistoryQueue(shape=self._image_shape, size=50)
        history_b = HistoryQueue(shape=self._image_shape, size=50)
        # TODO: infinite loop
        steps = trange(initial_step, num_global_step, total=num_global_step, initial=initial_step)
        return epoch_length, history_a, history_b, lr_decay, lr_initial, num_initial_iter, steps

    def should_write_summary(self, step):
        return step % self._log_step == 0

    def should_save_model(self, step):
        return step % self._save_step == 0

    def init_training_parameters(self, sess):
        # TODO: Replace hard-coded number, refactor, maybe think of infinity loop
        epoch_length = 1000  # min(len(data_A), len(data_B))
        num_batch = epoch_length // self._batch_size
        epoch_length = num_batch * self._batch_size
        num_initial_iter = 100
        num_decay_iter = 100
        lr_initial = 0.0001
        lr_decay = lr_initial / num_decay_iter
        initial_step = sess.run(self.placeholders.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        return epoch_length, initial_step, lr_decay, lr_initial, num_global_step, num_initial_iter

    def get_learning_rate(self, step, epoch_length, lr_decay, lr_initial, num_initial_iter):
        epoch = step // epoch_length
        if epoch > num_initial_iter:
            return max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)
        else:
            return lr_initial

    def get_fake_frames(self, frames_a, frames_b, sess):

        fake_a, fake_b = sess.run([self.images.frames_ba, self.images.frames_ab],
                                  feed_dict={self.placeholders.frames_a: frames_a,
                                             self.placeholders.frames_b: frames_b,
                                             self.placeholders.is_train: True})

        return fake_a, fake_b

    def get_fake_images(self, image_a, image_b, sess):
        fake_a, fake_b = sess.run([self.images.image_ba, self.images.image_ab],
                                  feed_dict={self.placeholders.image_a: image_a,
                                             self.placeholders.image_b: image_b,
                                             self.placeholders.is_train: True})
        return fake_a, fake_b

    def query_history_queue(self, fake_a, fake_b, history_a, history_b):
        fake_a = history_a.query(fake_a)
        fake_b = history_b.query(fake_b)
        return fake_a, fake_b

    def get_optical_flows(self, image_a, image_b, sess):
        flow_a = self.get_flow(image_a, sess)
        flow_b = self.get_flow(image_b, sess)

        return flow_a, flow_b

    def get_flow(self, image_series, sess):
        backwards, forwards = sess.run([self.networks.backwards_flow, self.networks.forwards_flow],
                                       feed_dict={self.placeholders.image_warp_input: image_series})
        return np.stack((backwards, forwards), axis=1)

    def get_warped_images(self, image_a, image_b, fake_a, fake_b, sess):
        warped_image_a, warped_fake_a = sess.run([self.networks.warped_real, self.networks.warped_fake],
                                                 feed_dict={self.placeholders.image_warp_input: image_a,
                                                            self.placeholders.fake_warp_input: fake_a})
        warped_image_b, warped_fake_b = sess.run([self.networks.warped_real, self.networks.warped_fake],
                                                 feed_dict={self.placeholders.image_warp_input: image_b,
                                                            self.placeholders.fake_warp_input: fake_b})

        return warped_image_a, warped_image_b, warped_fake_a, warped_fake_b

    def get_real_frames(self, data_A, data_B, sess):
        return self.get_dataset_sample(data_A, data_B, sess)

    def get_real_images(self, data_A, data_B, sess):
        image_a, image_b = self.get_dataset_sample(data_A, data_B, sess)
        return image_a[:, 0], image_b[:, 0]

    def get_dataset_sample(self, data_A, data_B, sess):
        image_a = sess.run(data_A)
        image_b = sess.run(data_B)
        return image_a, image_b

    def write_summary(self, fetched, step, steps, summary_writer):

        summary_writer.add_summary(fetched[-1], step)
        summary_writer.flush()
        steps.set_description(
            'Loss: D_a({:.3f}) D_b({:.3f}) D_temporal({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                fetched[0], fetched[1], fetched[5], fetched[2], fetched[3], fetched[5], fetched[4]))

    def get_fetches(self, step, video_training):
        fetches = []
        fetches = self.add_losses(fetches)
        fetches = self.add_generator_optimizer(fetches)
        fetches = self.add_discriminator_optimizer(fetches, video_training)
        fetches = self.add_summary(fetches, step)
        return fetches

    def add_losses(self, fetches):
        fetches += [self.losses.loss_D_a, self.losses.loss_D_b, self.losses.loss_G_spat_ab,
                    self.losses.loss_G_spat_ba, self.losses.loss_cycle, self.losses.loss_D_temp]
        return fetches

    def add_generator_optimizer(self, fetches):
        fetches += [self.optimizers.optimizer_G_ab, self.optimizers.optimizer_G_ba]
        return fetches

    def add_discriminator_optimizer(self, fetches, video_training):
        fetches += [self.optimizers.optimizer_D_a, self.optimizers.optimizer_D_b]
        if video_training:
            fetches += [self.optimizers.optimizer_D_temp, self.optimizers.optimizer_fnet]
        return fetches

    def add_summary(self, fetches, step):
        if self.should_write_summary(step):
            fetches += [self.tb_summary.summary_op]
        return fetches
