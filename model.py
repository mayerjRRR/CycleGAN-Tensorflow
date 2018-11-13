import os

from tqdm import trange
from scipy.misc import imsave
import tensorflow as tf
import numpy as np

from generator import Generator
from discriminator import Discriminator
from history_queue import HistoryQueue
from utils import logger


class CycleGAN(object):
    def __init__(self, args):
        self.init_parameters(args)

        history_fake_a, history_fake_b, image_a, image_b = self.init_placeholders()

        discriminator_a, discriminator_b, generator_ab, generator_ba = self.init_networks()

        image_ab, image_aba, image_ba, image_bab = self.generate_fake_images(generator_ab, generator_ba, image_a,
                                                                             image_b)

        D_fake_a, D_fake_b, D_history_fake_a, D_history_fake_b, D_real_a, D_real_b = self.discrimintate_images(
            discriminator_a, discriminator_b, history_fake_a, history_fake_b, image_a, image_ab, image_b, image_ba)

        loss_D_a, loss_D_b, loss_G_ab, loss_G_ab_final, loss_G_ba, loss_G_ba_final, loss_cycle = self.define_losses(
            D_fake_a, D_fake_b, D_history_fake_a, D_history_fake_b, D_real_a, D_real_b, image_a, image_aba, image_b,
            image_bab)

        self.init_optimizers(discriminator_a, discriminator_b, generator_ab, generator_ba, loss_D_a, loss_D_b,
                             loss_G_ab_final, loss_G_ba_final)
        self._build_tensorboard_summary(D_fake_a, D_fake_b, D_real_a, D_real_b, image_a, image_ab, image_aba, image_b,
                                        image_ba, image_bab, loss_D_a, loss_D_b, loss_G_ab, loss_G_ba, loss_cycle)

    def init_optimizers(self, discriminator_a, discriminator_b, generator_ab, generator_ba, loss_D_a, loss_D_b,
                        loss_G_ab_final, loss_G_ba_final):
        self.optimizer_D_a = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(loss_D_a, var_list=discriminator_a.var_list, global_step=self.global_step)
        self.optimizer_D_b = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(loss_D_b, var_list=discriminator_b.var_list)
        self.optimizer_G_ab = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(loss_G_ab_final, var_list=generator_ab.var_list)
        self.optimizer_G_ba = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(loss_G_ba_final, var_list=generator_ba.var_list)

    def define_losses(self, D_fake_a, D_fake_b, D_history_fake_a, D_history_fake_b, D_real_a, D_real_b, image_a,
                      image_aba, image_b, image_bab):
        # Least squre loss for GAN discriminator
        loss_D_a = (tf.reduce_mean(tf.squared_difference(D_real_a, 0.9)) +
                    tf.reduce_mean(tf.square(D_history_fake_a))) * 0.5
        loss_D_b = (tf.reduce_mean(tf.squared_difference(D_real_b, 0.9)) +
                    tf.reduce_mean(tf.square(D_history_fake_b))) * 0.5
        # Least squre loss for GAN generator
        loss_G_ab = tf.reduce_mean(tf.squared_difference(D_fake_b, 0.9))
        loss_G_ba = tf.reduce_mean(tf.squared_difference(D_fake_a, 0.9))
        # L1 norm for reconstruction error
        loss_rec_aba = tf.reduce_mean(tf.abs(image_a - image_aba))
        loss_rec_bab = tf.reduce_mean(tf.abs(image_b - image_bab))
        loss_cycle = self._cycle_loss_coeff * (loss_rec_aba + loss_rec_bab)
        loss_G_ab_final = loss_G_ab + loss_cycle
        loss_G_ba_final = loss_G_ba + loss_cycle
        return loss_D_a, loss_D_b, loss_G_ab, loss_G_ab_final, loss_G_ba, loss_G_ba_final, loss_cycle

    def discrimintate_images(self, discriminator_a, discriminator_b, history_fake_a, history_fake_b, image_a, image_ab,
                             image_b, image_ba):
        D_real_a = discriminator_a(image_a)
        D_fake_a = discriminator_a(image_ba)
        D_real_b = discriminator_b(image_b)
        D_fake_b = discriminator_b(image_ab)
        D_history_fake_a = discriminator_a(history_fake_a)
        D_history_fake_b = discriminator_b(history_fake_b)
        return D_fake_a, D_fake_b, D_history_fake_a, D_history_fake_b, D_real_a, D_real_b

    def generate_fake_images(self, generator_ab, generator_ba, image_a, image_b):
        image_ab = self.image_ab = generator_ab(image_a)
        image_ba = self.image_ba = generator_ba(image_b)
        image_bab = self.image_bab = generator_ab(image_ba)
        image_aba = self.image_aba = generator_ba(image_ab)
        return image_ab, image_aba, image_ba, image_bab

    def init_networks(self):
        generator_ab, generator_ba = self.init_generators()
        discriminator_a, discriminator_b = self.init_discriminators()
        return discriminator_a, discriminator_b, generator_ab, generator_ba

    def init_discriminators(self):
        discriminator_a = Discriminator('discriminator_a', is_train=self.is_train,
                                        norm='instance', activation='leaky')
        discriminator_b = Discriminator('discriminator_b', is_train=self.is_train,
                                        norm='instance', activation='leaky')
        return discriminator_a, discriminator_b

    def init_generators(self):
        generator_ab = Generator('generator_ab', is_train=self.is_train,
                                 norm='instance', activation='relu', image_size=self._image_size)
        generator_ba = Generator('generator_ba', is_train=self.is_train,
                                 norm='instance', activation='relu', image_size=self._image_size)
        return generator_ab, generator_ba

    def init_placeholders(self):
        self.init_training_placeholders()
        image_a, image_b = self.init_real_placeholders()
        history_fake_a, history_fake_b = self.init_fake_placeholders()
        image_a, image_b = self.augment_data_for_training(image_a, image_b)
        return history_fake_a, history_fake_b, image_a, image_b

    def init_parameters(self, args):
        self.init_args(args)
        self.init_image_dimensions()

    def init_image_dimensions(self):
        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15)
        self._image_shape = [self._image_size, self._image_size, 3]

    def init_args(self, args):
        self._log_step = args.log_step
        self._batch_size = args.batch_size
        self._image_size = args.image_size
        self._cycle_loss_coeff = args.cycle_loss_coeff

    def init_training_placeholders(self):
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.global_step = tf.contrib.framework.get_or_create_global_step(
            graph=None)  # Tensorflow magic global training step index

    def init_fake_placeholders(self):
        history_fake_a = self.history_fake_a = \
            tf.placeholder(tf.float32, [None] + self._image_shape, name='history_fake_a')
        history_fake_b = self.history_fake_b = \
            tf.placeholder(tf.float32, [None] + self._image_shape, name='history_fake_b')
        return history_fake_a, history_fake_b

    def init_real_placeholders(self):
        image_a = self.image_a = \
            tf.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_a')
        image_b = self.image_b = \
            tf.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_b')
        return image_a, image_b

    def augment_data_for_training(self, image_a, image_b):
        # Data augmentation
        def augment_image(image):
            image = tf.image.resize_images(image, [self._augment_size, self._augment_size])
            image = tf.random_crop(image, [self._batch_size] + self._image_shape)
            image = tf.map_fn(tf.image.random_flip_left_right, image)
            return image

        image_a = tf.cond(self.is_train,
                          lambda: augment_image(image_a),
                          lambda: image_a)
        image_b = tf.cond(self.is_train,
                          lambda: augment_image(image_b),
                          lambda: image_b)
        return image_a, image_b

    def _build_tensorboard_summary(self, D_fake_a, D_fake_b, D_real_a, D_real_b, image_a, image_ab, image_aba, image_b,
                                   image_ba, image_bab, loss_D_a, loss_D_b, loss_G_ab, loss_G_ba, loss_cycle):
        self.loss_D_a = loss_D_a
        self.loss_D_b = loss_D_b
        self.loss_G_ab = loss_G_ab
        self.loss_G_ba = loss_G_ba
        self.loss_cycle = loss_cycle
        tf.summary.scalar('loss/dis_A', loss_D_a)
        tf.summary.scalar('loss/dis_B', loss_D_b)
        tf.summary.scalar('loss/gen_AB', loss_G_ab)
        tf.summary.scalar('loss/gen_BA', loss_G_ba)
        tf.summary.scalar('loss/cycle', loss_cycle)
        tf.summary.scalar('model/D_a_real', tf.reduce_mean(D_real_a))
        tf.summary.scalar('model/D_a_fake', tf.reduce_mean(D_fake_a))
        tf.summary.scalar('model/D_b_real', tf.reduce_mean(D_real_b))
        tf.summary.scalar('model/D_b_fake', tf.reduce_mean(D_fake_b))
        tf.summary.scalar('model/lr', self.lr)
        tf.summary.image('A/A', image_a[0:1])
        tf.summary.image('A/A-B', image_ab[0:1])
        tf.summary.image('A/A-B-A', image_aba[0:1])
        tf.summary.image('B/B', image_b[0:1])
        tf.summary.image('B/B-A', image_ba[0:1])
        tf.summary.image('B/B-A-B', image_bab[0:1])
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, data_A, data_B):
        logger.info('Start training.')
#        logger.info('  {} images from A'.format(len(data_A)))
 #       logger.info('  {} images from B'.format(len(data_B)))

        data_size = 100000 #min(len(data_A), len(data_B))
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

        #TODO: infinite loop
        steps = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in steps:
            #TODO: resume training with global_step
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

            #TODO: replace by feedable iterator

            image_a = sess.run(data_A)
            image_b = sess.run(data_B)

            fake_a, fake_b = sess.run([self.image_ba, self.image_ab],
                                      feed_dict={self.image_a: image_a,
                                                self.image_b: image_b,
                                                 self.is_train: True})

            fake_a = history_a.query(fake_a)
            fake_b = history_b.query(fake_b)

            fetches = [self.loss_D_a, self.loss_D_b, self.loss_G_ab,
                       self.loss_G_ba, self.loss_cycle,
                       self.optimizer_D_a, self.optimizer_D_b,
                       self.optimizer_G_ab, self.optimizer_G_ba]
            if step % self._log_step == 0:
                fetches += [self.summary_op]

            fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.history_fake_a: fake_a,
                                                   self.history_fake_b: fake_b})

            if step % self._log_step == 0:
                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                steps.set_description(
                    'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                        fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))


    def test(self, sess, data_A, data_B, base_dir):
        #TODO: Implement Iterator/Dataset based solution
        step = 0
        for data in data_A:
            step += 1
            fetches = [self.image_ab, self.image_aba]
            image_a = np.expand_dims(data, axis=0)
            image_ab, image_aba = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                    self.is_train: False})
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
