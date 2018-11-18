import tensorflow as tf

from components.images import Images
from components.losses import Losses
from components.placeholders import Placeholders


class TensorBoardSummary:
    def __init__(self, images: Images, losses: Losses,
                 placeholders: Placeholders):

        tf.summary.scalar('loss/dis_A', losses.loss_D_a)
        tf.summary.scalar('loss/dis_B', losses.loss_D_b)
        tf.summary.scalar('loss/gen_AB', losses.loss_G_ab)
        tf.summary.scalar('loss/gen_BA', losses.loss_G_ba)
        tf.summary.scalar('loss/cycle', losses.loss_cycle)
        tf.summary.scalar('model/D_a_real', tf.reduce_mean(losses.D_real_a))
        tf.summary.scalar('model/D_a_fake', tf.reduce_mean(losses.D_fake_a))
        tf.summary.scalar('model/D_b_real', tf.reduce_mean(losses.D_real_b))
        tf.summary.scalar('model/D_b_fake', tf.reduce_mean(losses.D_fake_b))
        tf.summary.scalar('model/lr', placeholders.lr)
        tf.summary.image('A/A', images.image_a[0:1])
        tf.summary.image('A/A-B', images.image_ab[0:1])
        tf.summary.image('A/A-B-A', images.image_aba[0:1])
        tf.summary.image('B/B', images.image_b[0:1])
        tf.summary.image('B/B-A', images.image_ba[0:1])
        tf.summary.image('B/B-A-B', images.image_bab[0:1])
        self.summary_op = tf.summary.merge_all()