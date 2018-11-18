from components.placeholders import Placeholders
from nets.discriminator import Discriminator
from nets.generator import Generator


class Networks:
    def __init__(self, placeholders: Placeholders, image_size):
        self.init_generators(placeholders, image_size)
        self.init_discriminators(placeholders)

    def init_generators(self, placeholders: Placeholders, image_size):
        self.generator_ab = Generator('generator_ab', is_train=placeholders.is_train,
                                      norm='instance', activation='relu', image_size=image_size)
        self.generator_ba = Generator('generator_ba', is_train=placeholders.is_train,
                                      norm='instance', activation='relu', image_size=image_size)

    def init_discriminators(self, placeholders: Placeholders):
        self.discriminator_a = Discriminator('discriminator_a', is_train=placeholders.is_train,
                                             norm='instance', activation='leaky')
        self.discriminator_b = Discriminator('discriminator_b', is_train=placeholders.is_train,
                                             norm='instance', activation='leaky')