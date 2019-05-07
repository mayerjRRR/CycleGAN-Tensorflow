from tqdm import trange

from src.components.images import Images
from src.components.losses import Losses
from src.components.networks import Networks
from src.components.optimizers import Optimizers
from src.components.placeholders import Placeholders
from src.components.savers import Savers
from src.components.tensor_board_summary import TensorBoardSummary
from src.utils.history_queue import HistoryQueue
from src.utils.argument_parser import TrainingConfig


class CycleGan(object):

    def __init__(self, training_config: TrainingConfig, train_videos=True, train_images=False):
        self.init_parameters(training_config, train_videos, train_images)

        self.placeholders = Placeholders(self.config.batch_size, self.image_shape, self.input_shape, training_config.frame_sequence_length)
        self.networks = Networks(self.placeholders)
        self.images = Images(self.placeholders, self.networks, self.augmentation_shape, self.image_shape)
        self.losses = Losses(self.networks, self.placeholders, self.images, self.config, self.train_videos, self.train_images)
        self.optimizers = Optimizers(self.networks, self.losses, self.placeholders, self.train_videos)
        self.tb_summary = TensorBoardSummary(self.images, self.losses, self.placeholders, self.train_videos,
                                             self.train_images)
        self.savers = Savers(self.networks, self.placeholders, self.config.model_directory, self.config.initialization_model)

    def init_parameters(self, training_config: TrainingConfig, train_videos, train_images):
        self.config = training_config
        self.save_training_mode(train_videos,train_images)
        self.init_image_dimensions()


    def save_training_mode(self, train_videos, train_images):
        self.train_videos = train_videos
        self.train_images = train_images


    def init_image_dimensions(self):
        self.input_shape = [self.config.data_size, self.config.data_size, 3]
        self.augmentation_shape = [self.config.training_size + int(self.config.training_size / 8),
                                   self.config.training_size + int(self.config.training_size / 8)]
        self.image_shape = [self.config.training_size, self.config.training_size, 3]


    def train_on_videos(self, sess, summary_writer, frame_data_a, frame_data_b, learning_rate):
        self.init_training(sess,learning_rate)

        fake_a_history, fake_b_history = self.init_fake_frame_history(frame_data_a, frame_data_b, sess)

        for step in self.steps:
            lr = self.get_learning_rate(step)

            frames_a, frames_b = self.get_real_frames(frame_data_a, frame_data_b, sess)

            fetches = self.get_fetches(step, video_training=True)

            fetched = sess.run(fetches, feed_dict={self.placeholders.frames_a: frames_a,
                                                   self.placeholders.frames_b: frames_b,
                                                   self.placeholders.is_train: True,
                                                   self.placeholders.lr: lr,
                                                   self.placeholders.history_fake_temp_frames_a: fake_a_history,
                                                   self.placeholders.history_fake_temp_frames_b: fake_b_history})
            fake_a_history, fake_b_history = self.update_fake_frame_history(fetched)

            if self.should_write_summary(step):
                self.write_summary(fetched, step, summary_writer)
            if self.should_save_model(step):
                self.savers.save_all(sess, global_step=step)


    def init_fake_frame_history(self, frame_data_a, frame_data_b, sess):
        frames_a, frames_b = self.get_real_frames(frame_data_a, frame_data_b, sess)
        warped_fakes_a, warped_fakes_b = self.get_fake_warped_frames(frames_a, frames_b, sess)
        fake_a_history, fake_b_history = self.query_history_queue(warped_fakes_a, warped_fakes_b)
        return fake_a_history, fake_b_history


    def get_fake_warped_frames(self, frames_a, frames_b, sess):
        warped_fakes_a, warped_fakes_b = sess.run([self.images.warped_frames_ba, self.images.warped_frames_ab],
                                                  feed_dict={self.placeholders.frames_a: frames_a,
                                                             self.placeholders.frames_b: frames_b,
                                                             self.placeholders.is_train: True})
        return warped_fakes_a, warped_fakes_b


    def update_fake_frame_history(self, fetched):
        warped_fakes_a = fetched['fakes'][0]
        warped_fakes_b = fetched['fakes'][1]
        fake_a_history, fake_b_history = self.query_history_queue(warped_fakes_a, warped_fakes_b)
        return fake_a_history, fake_b_history


    def train_on_images(self, sess, summary_writer, image_data_a, image_data_b, learning_rate):
        self.init_training(sess,learning_rate)

        fake_a_history, fake_b_history = self.init_fake_image_history(image_data_a, image_data_b, sess)

        for step in self.steps:
            lr = self.get_learning_rate(step)

            image_a, image_b = self.get_real_images(image_data_a, image_data_b, sess)

            fetches = self.get_fetches(step, video_training=False)

            fetched = sess.run(fetches, feed_dict={self.placeholders.image_a: image_a,
                                                   self.placeholders.image_b: image_b,
                                                   self.placeholders.is_train: True,
                                                   self.placeholders.lr: lr,
                                                   self.placeholders.history_fake_a: fake_a_history,
                                                   self.placeholders.history_fake_b: fake_b_history})
            fake_a_history, fake_b_history = self.update_fake_image_history(fetched)
            if self.should_write_summary(step):
                self.write_summary(fetched, step, summary_writer)
            if self.should_save_model(step):
                self.savers.save_all(sess)


    def init_fake_image_history(self, image_data_a, image_data_b, sess):
        image_a, image_b = self.get_real_images(image_data_a, image_data_b, sess)
        fake_a, fake_b = self.get_fake_images(image_a, image_b, sess)
        fake_a_history, fake_b_history = self.query_history_queue(fake_a, fake_b)
        return fake_a_history, fake_b_history


    def update_fake_image_history(self, fetched):
        fake_a = fetched['fakes'][0]
        fake_b = fetched['fakes'][1]
        fake_a_history, fake_b_history = self.query_history_queue(fake_a, fake_b)
        return fake_a_history, fake_b_history


    def init_training(self, sess, learning_rate):
        self.epoch_length, initial_step, self.lr_decay, self.lr_initial, num_global_step, self.initial_iterations = \
            self.init_training_parameters(sess, learning_rate)
        self.history_queue_a = HistoryQueue(size=50)
        self.history_queue_b = HistoryQueue(size=50)
        self.steps = trange(initial_step, num_global_step, total=num_global_step, initial=initial_step)


    def should_write_summary(self, step):
        return step % self.config.logging_frequency == 0


    def should_save_model(self, step):
        return step % self.config.saving_frequency == 0


    def init_training_parameters(self, sess, learning_rate):
        epoch_length = 500
        num_initial_iter = 150
        num_decay_iter = 50
        lr_initial = learning_rate
        lr_decay = lr_initial / num_decay_iter
        initial_step = sess.run(self.placeholders.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        return epoch_length, initial_step, lr_decay, lr_initial, num_global_step, num_initial_iter


    def get_learning_rate(self, step):
        epoch = step // self.epoch_length
        if epoch > self.initial_iterations:
            return max(0.0, self.lr_initial - (epoch - self.initial_iterations) * self.lr_decay)
        else:
            return self.lr_initial


    def get_fake_images(self, image_a, image_b, sess):
        fake_a, fake_b = sess.run([self.images.image_ba, self.images.image_ab],
                                  feed_dict={self.placeholders.image_a: image_a,
                                             self.placeholders.image_b: image_b,
                                             self.placeholders.is_train: True})
        return fake_a, fake_b


    def query_history_queue(self, fake_a, fake_b):
        fake_a = self.history_queue_a.query(fake_a)
        fake_b = self.history_queue_b.query(fake_b)
        return fake_a, fake_b


    def get_real_frames(self, data_A, data_B, sess):
        return self.get_dataset_sample(data_A, data_B, sess)


    def get_real_images(self, data_A, data_B, sess):
        image_a, image_b = self.get_dataset_sample(data_A, data_B, sess)
        return image_a[:, 0], image_b[:, 0]


    def get_dataset_sample(self, data_A, data_B, sess):
        image_a = sess.run(data_A)
        image_b = sess.run(data_B)
        return image_a, image_b


    def write_summary(self, fetched, step, summary_writer):
        summary_writer.add_summary(fetched['summary'], step)
        summary_writer.flush()
        losses = fetched['losses']
        self.steps.set_description(
            'Loss: D_a({:.3f}) D_b({:.3f}) D_temporal({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f}) identity({:.3f}) code({:.3f})'.format(
                losses[0], losses[1], losses[5], losses[2], losses[3], losses[4], losses[6], losses[7]))


    def get_fetches(self, step, video_training):
        fetches = {}
        fetches = self.add_losses(fetches)
        fetches = self.add_generator_optimizer(fetches)
        fetches = self.add_discriminator_optimizer(fetches, video_training)
        fetches = self.add_fakes(fetches, video_training)
        fetches = self.add_summary(fetches, step)
        return fetches


    def add_losses(self, fetches):
        fetches['losses'] = [self.losses.loss_D_a, self.losses.loss_D_b, self.losses.loss_G_spat_ab,
                             self.losses.loss_G_spat_ba, self.losses.loss_cycle, self.losses.loss_D_temp, self.losses.loss_identity, self.losses.loss_code]
        return fetches


    def add_generator_optimizer(self, fetches):
        fetches['generator_optimizer'] = [self.optimizers.optimizer_G_ab, self.optimizers.optimizer_G_ba]
        return fetches


    def add_discriminator_optimizer(self, fetches, video_training):
        fetches['discriminator_optimizer'] = [self.optimizers.optimizer_D_a, self.optimizers.optimizer_D_b]
        if video_training:
            fetches['discriminator_optimizer'] += [self.optimizers.optimizer_D_temp]
        return fetches


    def add_summary(self, fetches, step):
        if self.should_write_summary(step):
            fetches['summary'] = self.tb_summary.summary_op
        return fetches


    def add_fakes(self, fetches, video_training):
        if video_training:
            fetches['fakes'] = [self.images.warped_frames_ba, self.images.warped_frames_ab]
        else:
            fetches['fakes'] = [self.images.image_ba, self.images.image_ab]
        return fetches
