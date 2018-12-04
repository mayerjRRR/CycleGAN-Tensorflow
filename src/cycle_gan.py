import os

from tqdm import trange
from scipy.misc import imsave
import numpy as np

from src.components.images import Images
from src.components.losses import Losses
from src.components.networks import Networks
from src.components.optimizers import Optimizers
from src.components.placeholders import Placeholders
from src.components.tensor_board_summary import TensorBoardSummary
from src.utils.history_queue import HistoryQueue
from src.utils.utils import logger


class CycleGan(object):

    def __init__(self, image_size=256, batch_size=4, cycle_loss_coeff=1, log_step=10):
        self.init_parameters(image_size, batch_size, cycle_loss_coeff,log_step)

        self.placeholders = Placeholders(self._batch_size, self._image_shape)
        self.networks = Networks(self.placeholders, self._image_size)
        self.images = Images(self.placeholders, self.networks, self._image_shape, self._batch_size, self._augment_size)
        self.losses = Losses(self.networks, self.placeholders, self.images, self._cycle_loss_coeff)
        self.optimizers = Optimizers(self.networks, self.losses, self.placeholders)
        self.tb_summary = TensorBoardSummary(self.images, self.losses, self.placeholders)

    def init_parameters(self, image_size, batch_size, cycle_loss_coeff, log_step):
        self.init_args(image_size, batch_size, cycle_loss_coeff, log_step)
        self.init_image_dimensions()

    def init_args(self, image_size, batch_size, cycle_loss_coeff, log_step):
        self._log_step = log_step
        self._batch_size = batch_size
        self._image_size = image_size
        self._cycle_loss_coeff = cycle_loss_coeff

    def init_image_dimensions(self):
        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15)
        self._image_shape = [self._image_size, self._image_size, 3]



    def train(self, sess, summary_writer, data_A, data_B):
        logger.info('Start training.')

        epoch_length, initial_step, lr_decay, lr_initial, num_global_step, num_initial_iter = \
            self.init_training_parameters(sess)

        history_a = HistoryQueue(shape=self._image_shape, size=50)
        history_b = HistoryQueue(shape=self._image_shape, size=50)

        # TODO: infinite loop
        steps = trange(initial_step, num_global_step, total=num_global_step, initial=initial_step)
        for step in steps:
            epoch = step // epoch_length
            lr = self.get_learning_rate(epoch, lr_decay, lr_initial, num_initial_iter)

            image_a, image_b = self.get_real_images(data_A, data_B, sess)
            fake_a, fake_b = self.get_fake_images(history_a, history_b, image_a, image_b, sess)

            fetches = self.get_fetches(step)

            fetched = sess.run(fetches, feed_dict={self.placeholders.image_a: image_a,
                                                   self.placeholders.image_b: image_b,
                                                   self.placeholders.is_train: True,
                                                   self.placeholders.lr: lr,
                                                   self.placeholders.history_fake_a_placeholder: fake_a,
                                                   self.placeholders.history_fake_b_placeholder: fake_b})

            self.write_summary(fetched, step, steps, summary_writer)

    def init_training_parameters(self, sess):
        # TODO: Replace hard-coded number, refactor, maybe think of infinity loop
        epoch_length = 1000  # min(len(data_A), len(data_B))
        num_batch = epoch_length // self._batch_size
        epoch_length = num_batch * self._batch_size
        num_initial_iter = 100
        num_decay_iter = 100
        lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter
        initial_step = sess.run(self.placeholders.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        return epoch_length, initial_step, lr_decay, lr_initial, num_global_step, num_initial_iter

    def get_learning_rate(self, epoch, lr_decay, lr_initial, num_initial_iter):
        if epoch > num_initial_iter:
            return max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)
        else:
            return lr_initial

    def get_fake_images(self, history_a, history_b, image_a, image_b, sess):
        fake_a, fake_b = sess.run([self.images.image_ba, self.images.image_ab],
                                  feed_dict={self.placeholders.image_a: image_a,
                                             self.placeholders.image_b: image_b,
                                             self.placeholders.is_train: True})
        fake_a = history_a.query(fake_a)
        fake_b = history_b.query(fake_b)
        return fake_a, fake_b

    def get_real_images(self, data_A, data_B, sess):
        image_a = sess.run(data_A)[:,0]
        image_b = sess.run(data_B)[:,0]
        return image_a, image_b

    def write_summary(self, fetched, step, steps, summary_writer):
        if step % self._log_step == 0:
            summary_writer.add_summary(fetched[-1], step)
            summary_writer.flush()
            steps.set_description(
                'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                    fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))

    def get_fetches(self, step):
        fetches = []
        fetches = self.add_losses(fetches)
        fetches = self.add_generator_optimizer(fetches)
        fetches = self.add_discriminator_optimizer(fetches, step)
        fetches = self.add_summary(fetches, step)
        return fetches

    def add_losses(self, fetches):
        fetches += [self.losses.loss_D_a, self.losses.loss_D_b, self.losses.loss_G_ab,
                    self.losses.loss_G_ba, self.losses.loss_cycle]
        return fetches

    def add_generator_optimizer(self, fetches):
        fetches += [self.optimizers.optimizer_G_ab, self.optimizers.optimizer_G_ba]
        return fetches

    def add_discriminator_optimizer(self, fetches, step):
       # if step % 2 == 0:
        fetches += [self.optimizers.optimizer_D_a, self.optimizers.optimizer_D_b]
        return fetches

    def add_summary(self, fetches, step):
        if step % self._log_step == 0:
            fetches += [self.tb_summary.summary_op]
        return fetches

    def test(self, sess, data_A, data_B, base_dir):
        # TODO: Implement Iterator/Dataset based solution
        step = 0
        for data in data_A:
            step += 1
            fetches = [self.images.image_ab, self.images.image_aba]
            image_a = np.expand_dims(data, axis=0)
            image_ab, image_aba = sess.run(fetches, feed_dict={self.placeholders.image_a: image_a,
                                                               self.placeholders.is_train: False})
            images = np.concatenate((image_a, image_ab, image_aba), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'a_to_b_{}.jpg'.format(step)), images)

        step = 0
        for data in data_B:
            step += 1
            fetches = [self.image_ba, self.image_bab]
            image_b = np.expand_dims(data, axis=0)
            image_ba, image_bab = sess.run(fetches, feed_dict={self.image_b: image_b,
                                                               self.is_train: False})
            images = np.concatenate((image_b, image_ba, image_bab), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'b_to_a_{}.jpg'.format(step)), images)
