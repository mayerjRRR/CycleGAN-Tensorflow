import tensorflow as tf

from src.components.images import Images
from src.components.losses import Losses
from src.components.placeholders import Placeholders


class TensorBoardSummary:

    def __init__(self, images: Images, losses: Losses,
                 placeholders: Placeholders, train_videos, train_images):

        self.add_losses(losses, train_videos)
        self.add_spacial_discriminator_outputs(losses, train_images, train_videos)
        self.add_temporal_discriminator_outputs(losses, train_videos)
        self.add_model_parameters(losses, placeholders, train_videos)
        self.add_images(images, train_images, train_videos)
        self.summary_op = tf.summary.merge_all()

    def add_losses(self, losses, train_videos):
        tf.summary.scalar('Losses/Spacial_Discriminator_A', losses.loss_D_a)
        tf.summary.scalar('Losses/Spacial_Discriminator_B', losses.loss_D_b)
        if train_videos:
            tf.summary.scalar('Losses/Temporal_Discriminator', losses.loss_D_temp)
        tf.summary.scalar('Losses/Generator_AB', losses.loss_G_spat_ab)
        tf.summary.scalar('Losses/Generator_BA', losses.loss_G_spat_ba)
        tf.summary.scalar('Losses/Cycle_Loss', losses.loss_cycle)

    def add_spacial_discriminator_outputs(self, losses, train_images, train_videos):
        if train_videos:
            tf.summary.scalar('Spacial_Discriminator_Output/Real_Frame_A', tf.reduce_mean(losses.D_real_frame_a))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_Frame_A', tf.reduce_mean(losses.D_fake_frame_a))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_History_Frame_A',
                              tf.reduce_mean(losses.D_history_fake_frame_a))
            tf.summary.scalar('Spacial_Discriminator_Output/Real_Frame_B', tf.reduce_mean(losses.D_real_frame_b))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_Frame_B', tf.reduce_mean(losses.D_fake_frame_b))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_History_Frame_B',
                              tf.reduce_mean(losses.D_history_fake_frame_b))
        if train_images:
            tf.summary.scalar('Spacial_Discriminator_Output/Real_Image_A', tf.reduce_mean(losses.D_real_image_a))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_Image_A', tf.reduce_mean(losses.D_fake_image_a))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_History_Image_A',
                              tf.reduce_mean(losses.D_history_fake_image_a))
            tf.summary.scalar('Spacial_Discriminator_Output/Real_Image_B', tf.reduce_mean(losses.D_real_image_b))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_Image_B', tf.reduce_mean(losses.D_fake_image_b))
            tf.summary.scalar('Spacial_Discriminator_Output/Fake_History_Image_B',
                              tf.reduce_mean(losses.D_history_fake_image_b))

    def add_temporal_discriminator_outputs(self, losses, train_videos):
        if train_videos:
            tf.summary.scalar('Temporal_Discriminator_Output/Real_Frames_A', tf.reduce_mean(losses.D_temp_real_a))
            tf.summary.scalar('Temporal_Discriminator_Output/Fake_Frames_A', tf.reduce_mean(losses.D_temp_fake_a))
            tf.summary.scalar('Temporal_Discriminator_Output/Fake_Frames_from_History_A',
                              tf.reduce_mean(losses.D_temp_history_fake_a))
            tf.summary.scalar('Temporal_Discriminator_Output/Real_Frames_B', tf.reduce_mean(losses.D_temp_real_b))
            tf.summary.scalar('Temporal_Discriminator_Output/Fake_Frames_B', tf.reduce_mean(losses.D_temp_fake_b))
            tf.summary.scalar('Temporal_Discriminator_Output/Fake_Frames_from_History_B',
                              tf.reduce_mean(losses.D_temp_history_fake_b))

    def add_model_parameters(self, losses, placeholders, train_videos):
        if train_videos:
            tf.summary.scalar('Model_Parameters/Temporal_Loss_Weight', losses.temp_loss_weigth)
        tf.summary.scalar('Model_Parameters/Learning_Rate', placeholders.lr)

    def add_images(self, images, train_images, train_videos):
        if train_videos:
            self.add_a_frames(images)
            self.add_b_frames(images)

        if train_images:
            self.add_a_images(images)
            self.add_b_images(images)

    def add_a_frames(self, images):
        self.add_input_frames(images)
        self.add_warped_input_frames(images)
        self.add_fake_frames(images)
        self.add_warped_fake_frames(images)
        self.add_cycle_frames(images)

    def add_input_frames(self, images):
        tf.summary.image('A/Previous', images.frames_a[0:1, 0])
        tf.summary.image('A/Current', images.frames_a[0:1, 1])
        tf.summary.image('A/Next', images.frames_a[0:1, 2])
        tf.summary.image('A/Previous_Diff', tf.abs(images.frames_a[0:1, 1] - images.frames_a[0:1, 0]))
        tf.summary.image('A/Next_Diff', tf.abs(images.frames_a[0:1, 1] - images.frames_a[0:1, 2]))

    def add_warped_input_frames(self, images):
        tf.summary.image('A_Warped/Previous', images.warped_frames_a[0:1, 0])
        tf.summary.image('A_Warped/Current', images.warped_frames_a[0:1, 1])
        tf.summary.image('A_Warped/Next', images.warped_frames_a[0:1, 2])
        tf.summary.image('A_Warped/Previous_Diff',
                         tf.abs(images.warped_frames_a[0:1, 1] - images.warped_frames_a[0:1, 0]))
        tf.summary.image('A_Warped/Next_Diff',
                         tf.abs(images.warped_frames_a[0:1, 1] - images.warped_frames_a[0:1, 2]))

    def add_fake_frames(self, images):
        tf.summary.image('AB/Previous', images.frames_ab[0:1, 0])
        tf.summary.image('AB/Current', images.frames_ab[0:1, 1])
        tf.summary.image('AB/Next', images.frames_ab[0:1, 2])
        tf.summary.image('AB/Previous_Diff', tf.abs(images.frames_ab[0:1, 1] - images.frames_ab[0:1, 0]))
        tf.summary.image('AB/Next_Diff', tf.abs(images.frames_ab[0:1, 1] - images.frames_ab[0:1, 2]))

    def add_warped_fake_frames(self, images):
        tf.summary.image('AB_Warped/Previous', images.warped_frames_ab[0:1, 0])
        tf.summary.image('AB_Warped/Current', images.warped_frames_ab[0:1, 1])
        tf.summary.image('AB_Warped/Next', images.warped_frames_ab[0:1, 2])
        tf.summary.image('AB_Warped/Previous_Diff',
                         tf.abs(images.warped_frames_ab[0:1, 1] - images.warped_frames_ab[0:1, 0]))
        tf.summary.image('AB_Warped/Next_Diff',
                         tf.abs(images.warped_frames_ab[0:1, 1] - images.warped_frames_ab[0:1, 2]))

    def add_cycle_frames(self, images):
        tf.summary.image('ABA/Previous', images.frames_aba[0:1, 0])
        tf.summary.image('ABA/Current', images.frames_aba[0:1, 1])
        tf.summary.image('ABA/Next', images.frames_aba[0:1, 2])
        tf.summary.image('ABA/Previous_Diff', tf.abs(images.frames_aba[0:1, 1] - images.frames_aba[0:1, 0]))
        tf.summary.image('ABA/Next_Diff', tf.abs(images.frames_aba[0:1, 1] - images.frames_aba[0:1, 2]))

    def add_b_frames(self, images):
        tf.summary.image('B/B', images.warped_frames_b[0:1, 1])
        tf.summary.image('B/B-A', images.frames_ba[0:1, 1])
        tf.summary.image('B/B-A-B', images.frames_bab[0:1, 1])

    def add_b_images(self, images):
        tf.summary.image('B/B', images.image_b[0:1])
        tf.summary.image('B/B-A', images.image_ba[0:1])
        tf.summary.image('B/B-A-B', images.image_bab[0:1])

    def add_a_images(self, images):
        tf.summary.image('A/A', images.image_a[0:1])
        tf.summary.image('A/A-B', images.image_ab[0:1])
        tf.summary.image('A/A-B-A', images.image_aba[0:1])
