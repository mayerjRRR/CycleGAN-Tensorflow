import tensorflow as tf

from src.components.images import Images
from src.components.losses import Losses
from src.components.placeholders import Placeholders


class TensorBoardSummary:
    def __init__(self, images: Images, losses: Losses,
                 placeholders: Placeholders, train_videos, train_images):
        #TODO: implement for images
        tf.summary.scalar('loss/dis_A', losses.loss_D_a)
        tf.summary.scalar('loss/dis_B', losses.loss_D_b)
        if(train_videos):
            tf.summary.scalar('loss/dis_temp', losses.loss_D_temp)
        tf.summary.scalar('loss/gen_AB', losses.loss_G_spat_ab)
        tf.summary.scalar('loss/gen_BA', losses.loss_G_spat_ba)
        tf.summary.scalar('loss/cycle', losses.loss_cycle)
        #TODO: track discriminator output individually
        if (train_videos):
            tf.summary.scalar('model/D_frame_a_real', tf.reduce_mean(losses.D_real_frame_a))
            tf.summary.scalar('model/D_frame_a_fake', tf.reduce_mean(losses.D_fake_frame_a))
            tf.summary.scalar('model/D_frame_b_real', tf.reduce_mean(losses.D_real_frame_b))
            tf.summary.scalar('model/D_frame_b_fake', tf.reduce_mean(losses.D_fake_frame_b))

            tf.summary.scalar('model/D_temp_a_real', tf.reduce_mean(losses.D_temp_real_a))
            tf.summary.scalar('model/D_temp_a_fake', tf.reduce_mean(losses.D_temp_fake_a))
            tf.summary.scalar('model/D_temp_b_real', tf.reduce_mean(losses.D_temp_real_b))
            tf.summary.scalar('model/D_temp_b_fake', tf.reduce_mean(losses.D_temp_fake_b))
        if (train_images):
            tf.summary.scalar('model/D_image_a_real', tf.reduce_mean(losses.D_real_image_a))
            tf.summary.scalar('model/D_image_a_fake', tf.reduce_mean(losses.D_fake_image_a))
            tf.summary.scalar('model/D_image_b_real', tf.reduce_mean(losses.D_real_image_b))
            tf.summary.scalar('model/D_image_b_fake', tf.reduce_mean(losses.D_fake_image_b))

        tf.summary.scalar('model/lr', placeholders.lr)
        if (train_videos):
            tf.summary.image('A/Previous', images.frames_a[0:1,0])
            tf.summary.image('A/Current', images.frames_a[0:1,1])
            tf.summary.image('A/Next', images.frames_a[0:1,2])

            tf.summary.image('A_Warped/Previous', images.warped_frames_a[0:1,0])
            tf.summary.image('A_Warped/Current', images.warped_frames_a[0:1,1])
            tf.summary.image('A_Warped/Next', images.warped_frames_a[0:1,2])

            tf.summary.image('AB/Previous', images.frames_ab[0:1,0])
            tf.summary.image('AB/Current', images.frames_ab[0:1,1])
            tf.summary.image('AB/Next', images.frames_ab[0:1,2])

            tf.summary.image('AB_Warped/Previous', images.warped_frames_ab[0:1,0])
            tf.summary.image('AB_Warped/Current', images.warped_frames_ab[0:1,1])
            tf.summary.image('AB_Warped/Next', images.warped_frames_ab[0:1,2])

            tf.summary.image('ABA/Previous', images.frames_aba[0:1,0])
            tf.summary.image('ABA/Current', images.frames_aba[0:1,1])
            tf.summary.image('ABA/Next', images.frames_aba[0:1,2])

            tf.summary.image('B/B', images.warped_frames_b[0:1,1])
            tf.summary.image('B/B-A', images.frames_ba[0:1,1])
            tf.summary.image('B/B-A-B', images.frames_bab[0:1,1])
        if (train_images):
            tf.summary.image('A/A', images.image_a[0:1])
            tf.summary.image('A/A-B', images.image_ab[0:1])
            tf.summary.image('S/A-B-A', images.image_aba[0:1])

            tf.summary.image('B/B', images.image_b[0:1])
            tf.summary.image('B/B-A', images.image_ba[0:1])
            tf.summary.image('B/B-A-B', images.image_bab[0:1])
        self.summary_op = tf.summary.merge_all()