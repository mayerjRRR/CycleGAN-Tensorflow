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
        self.discriminator = Discriminator('discriminator', is_train=placeholders.is_train,
                                                     norm='instance', activation='leaky')
        '''
        outputs a vector of 4 probabilies for realA, fakeA, realB, fakeB respectively
        '''

