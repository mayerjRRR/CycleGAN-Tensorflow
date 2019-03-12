import os.path

import tensorflow as tf

from src.components.networks import Networks
from src.components.placeholders import Placeholders


class Saver:
    def __init__(self, variable_list, save_path, init_path=None, name="Unnamed Graph"):
        self.saver = tf.train.Saver(variable_list)
        self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.init_path = init_path
        self.name = name

    def load(self, session):
        try:
            self.saver.restore(session, tf.train.latest_checkpoint(self.save_path))
            print(f"Successfully restored {self.name} from {tf.train.latest_checkpoint(self.save_path)}!")
        except:
            print(f"Could not restore {self.name} from {self.save_path}. Maybe it doesn't exist!")
            self.attempt_restoring_from_initialization_checkpoint(session)

    def attempt_restoring_from_initialization_checkpoint(self, session):
        if self.init_path:
            print(f"Trying to initialize {self.name} from {self.init_path}.")
            try:
                self.saver.restore(session, tf.train.latest_checkpoint(self.init_path))
                print(f"Successfully restored {self.name} from {tf.train.latest_checkpoint(self.init_path)}!")
            except:
                print(f"Could not initialize {self.name} from {self.init_path}. Maybe it doesn't exist!")

    def save(self, session, global_step):
        self.saver.save(session, os.path.join(self.save_path, "model.ckpt"), global_step=global_step,
                        write_meta_graph=False)


class Savers:
    def __init__(self, networks: Networks, placeholders: Placeholders, save_dir, init_dir=None):
        self.save_dir = save_dir
        self.init_dir = init_dir

        self.init_generator_savers(networks)
        self.init_discriminator_savers(networks)
        self.init_fnet_saver()
        self.init_global_step_saver(placeholders)

    def init_fnet_saver(self):
        fnet_variable_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='fnet')
        self.fnet_saver = Saver(fnet_variable_list, save_path=self.get_save_path("fnet"), init_path='./fnet',
                                name="FNet")

    def init_discriminator_savers(self, networks):
        self.discriminator_spatial_a_saver = Saver(networks.discriminator_spatial_a.var_list,
                                                   save_path=self.get_save_path(networks.discriminator_spatial_a.name),
                                                   init_path=self.get_init_path(networks.discriminator_spatial_a.name),
                                                   name="Spacial Discriminator A")
        self.discriminator_spatial_b_saver = Saver(networks.discriminator_spatial_b.var_list,
                                                   save_path=self.get_save_path(networks.discriminator_spatial_b.name),
                                                   init_path=self.get_init_path(networks.discriminator_spatial_b.name),
                                                   name="Spacial Discriminator B")
        self.discriminator_temporal_saver = Saver(networks.discriminator_temporal.var_list,
                                                  save_path=self.get_save_path(networks.discriminator_temporal.name),
                                                  init_path=self.get_init_path(networks.discriminator_temporal.name),
                                                  name="Temporal Discriminator")

    def init_generator_savers(self, networks):
        self.generator_ab_saver = Saver(networks.generator_ab.var_list,
                                        save_path=self.get_save_path(networks.generator_ab.name),
                                        init_path=self.get_init_path(networks.generator_ab.name), name="Generator AB")
        self.generator_ba_saver = Saver(networks.generator_ba.var_list,
                                        save_path=self.get_save_path(networks.generator_ba.name),
                                        init_path=self.get_init_path(networks.generator_ba.name), name="Generator BA")

    def init_global_step_saver(self, placeholders):
        self.global_step_saver = Saver([placeholders.global_step], os.path.join(self.save_dir, "global_step"),
                                       name="Global Step")

    def save_all(self, session, global_step=None):
        for saver in self.get_all_savers():
            saver.save(session, global_step)

    def load_all(self, session):
        for saver in self.get_all_savers():
            saver.load(session)

    def get_all_savers(self):
        return [self.fnet_saver, self.discriminator_temporal_saver, self.discriminator_spatial_b_saver,
                self.discriminator_spatial_a_saver, self.generator_ab_saver, self.generator_ba_saver,
                self.global_step_saver]

    def get_init_path(self, name):
        return os.path.join(self.init_dir, name)

    def get_save_path(self, name):
        return os.path.join(self.save_dir, name)
