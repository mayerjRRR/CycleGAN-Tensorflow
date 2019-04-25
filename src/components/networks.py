from src.components.placeholders import Placeholders
from src.nets.discriminator import Discriminator
from src.nets.generator import Generator


class Networks:
    def __init__(self, placeholders: Placeholders):
        self.init_generators(placeholders)
        self.init_discriminators(placeholders)

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

        #self.discriminator_temporal = Discriminator('discriminator_temp', is_train=placeholders.is_train,
       #                                             norm='instance', activation='leaky')

