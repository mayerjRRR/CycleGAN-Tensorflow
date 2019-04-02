from src.components.placeholders import Placeholders
from src.nets.discriminator import Discriminator
from src.nets.generator import Generator

from src.utils.warp_utils import warp_to_middle_frame, get_flows_to_middle_frame


class Networks:
    def __init__(self, placeholders: Placeholders):
        self.init_generators(placeholders)
        self.init_discriminators(placeholders)
        self.init_fnet(placeholders)

    def init_generators(self, placeholders: Placeholders):
        self.generator_ab = Generator('generator_ab', is_train=placeholders.is_train, norm='instance',
                                      activation='relu')
        self.generator_ba = Generator('generator_ba', is_train=placeholders.is_train, norm='instance',
                                      activation='relu')

    def init_discriminators(self, placeholders: Placeholders):
        self.discriminator_spatial_a = Discriminator('discriminator_a', is_train=placeholders.is_train,
                                                     norm='instance', activation='leaky')
        self.discriminator_spatial_b = Discriminator('discriminator_b', is_train=placeholders.is_train,
                                                     norm='instance', activation='leaky')

        self.discriminator_temporal = Discriminator('discriminator_temp', is_train=placeholders.is_train,
                                                    norm='instance', activation='leaky')

    def init_fnet(self, placeholders: Placeholders):
        flows = get_flows_to_middle_frame(placeholders.image_warp_input)
        self.warped_real = warp_to_middle_frame(placeholders.image_warp_input, flows)
        self.warped_fake = warp_to_middle_frame(placeholders.fake_warp_input, flows)
