import os

from tqdm import trange
from scipy.misc import imsave
import numpy as np

from components.images import Images
from components.losses import Losses
from components.networks import Networks
from components.optimizers import Optimizers
from components.placeholders import Placeholders
from components.tensor_board_summary import TensorBoardSummary
from utils.history_queue import HistoryQueue
from utils.utils import logger


class CycleGan(object):

    def __init__(self, args):
        self.init_parameters(args)

        self.placeholders = Placeholders(self._batch_size, self._image_shape)
        self.networks = Networks(self.placeholders, self._image_size)
        self.images = Images(self.placeholders, self.networks, self._image_shape, self._batch_size, self._augment_size)
        self.losses = Losses(self.networks, self.placeholders, self.images, self._cycle_loss_coeff)
        self.optimizers = Optimizers(self.networks, self.losses, self.placeholders)
        self.tb_summary = TensorBoardSummary(self.images, self.losses, self.placeholders)

    def init_parameters(self, args):
        self.init_args(args)
        self.init_image_dimensions()

    def init_args(self, args):
        self._log_step = args.log_step
        self._batch_size = args.batch_size
        self._image_size = args.image_size
        self._cycle_loss_coeff = args.cycle_loss_coeff

    def init_image_dimensions(self):
        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15)
        self._image_shape = [self._image_size, self._image_size, 3]



    def train(self, sess, summary_writer, data_A, data_B):
        logger.info('Start training.')

    #TODO: Replace hard-coded number, refactor
        data_size = 100000  # min(len(data_A), len(data_B))
        num_batch = data_size // self._batch_size
        epoch_length = num_batch * self._batch_size

        num_initial_iter = 100
        num_decay_iter = 100
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter

        history_a = HistoryQueue(shape=self._image_shape, size=50)
        history_b = HistoryQueue(shape=self._image_shape, size=50)

        initial_step = sess.run(self.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length

        # TODO: infinite loop
        steps = trange(initial_step, num_global_step,
                       total=num_global_step, initial=initial_step)

        for step in steps:
            # TODO: resume training with global_step
            epoch = step // epoch_length
            iter = step % epoch_length

            if epoch > num_initial_iter:
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)

            #  if iter == 0:
            #     random.shuffle(data_A)
            #    random.shuffle(data_B)

            # get batches

            # next_element = iterator.get_next()
            # with tf.Session() as sess:
            # cv2.imshow("piff",sess.run(next_element))

            # image_a = np.stack(data_A[iter*self._batch_size:(iter+1)*self._batch_size])
            #  image_b = np.stack(data_B[iter*self._batch_size:(iter+1)*self._batch_size])

            # TODO: replace by feedable iterator or maybe not

            image_a = sess.run(data_A)
            image_b = sess.run(data_B)

            fake_a, fake_b = sess.run([self.images.image_ba, self.images.image_ab],
                                      feed_dict={self.placeholders.image_a: image_a,
                                                 self.placeholders.image_b: image_b,
                                                 self.placeholders.is_train: True})

            fake_a = history_a.query(fake_a)
            fake_b = history_b.query(fake_b)

            fetches = [self.losses.loss_D_a, self.losses.loss_D_b, self.losses.loss_G_ab,
                       self.losses.loss_G_ba, self.losses.loss_cycle,
                       self.optimizers.optimizer_D_a, self.optimizers.optimizer_D_b,
                       self.optimizers.optimizer_G_ab, self.optimizers.optimizer_G_ba]
            if step % self._log_step == 0:
                fetches += [self.tb_summary.summary_op]

            fetched = sess.run(fetches, feed_dict={self.placeholders.image_a: image_a,
                                                   self.placeholders.image_b: image_b,
                                                   self.placeholders.is_train: True,
                                                   self.placeholders.lr: lr,
                                                   self.placeholders.history_fake_a_placeholder: fake_a,
                                                   self.placeholders.history_fake_b_placeholder: fake_b})

            if step % self._log_step == 0:
                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                steps.set_description(
                    'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                        fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))

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
