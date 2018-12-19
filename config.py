# /usr/bin/python3.6

import keras.optimizers as opt
import keras.callbacks as kc
import os

class Config(object):

    def __init__(self):
        # data path
        self.root_path = os.path.realpath(__file__)

        # learning configurations
        # optimizer
        self.opt = 'adam'
        # learning rate
        self.lr = 3e-4
        # batch size
        self.batch_size = 4
        # epochs
        self.epochs = 40
        # validation
        self.validation_split = 0.05
        # weights_name
        self.weights_name = ''
        # log dir
        self.log_dir = ''

        # learning callbacks configurations
        self.use_earlystopping = True
        self.earlystopping_config = {'monitor': 'val_loss', 'patience': 50}
        self.use_reduceLR = True
        self.reduceLR_config = {'monitor':'loss', 'factor':0.4, 'patience':2,
                                'min_delta':1e-2, 'cooldown':0}
        self.use_checkpoint = True
        self.checkpoint_config = {'monitor':'val_loss', 'save_best_only':True,
                                  'save_weights_only':True}
        self.use_tensorborad = False
        self.tensorborad_config = {'histogram_freq': 2, 'write_images': True, 'update_freq': 1000}

    @property
    def optimizer(self):
        if self.opt == 'adam':
            return opt.Adam(self.lr)

    @property
    def learning_callbacks(self):
        callbacks = []
        # early stopping
        if self.use_earlystopping:
            callbacks.append(self._set_params(kc.EarlyStopping(),
                                              self.earlystopping_config))
        # reduce learning rate
        if self.use_reduceLR:
            callbacks.append(self._set_params(kc.ReduceLROnPlateau(),
                                              self.reduceLR_config))
        # check point
        if self.use_checkpoint:
            callbacks.append(self._set_params(kc.ModelCheckpoint(self.weights_name),
                                              self.checkpoint_config))
        # tensorborad
        if self.use_tensorborad:
            callbacks.append(self._set_params(kc.TensorBoard(self.log_dir),
                                              self.tensorborad_config))

        return callbacks

    def _set_params(self, obj, params):
        if hasattr(obj, '__setattr__'):
            for param_name in params:
                if hasattr(obj, param_name):
                    obj.__setattr__(param_name, params[param_name])
                else:
                    raise ValueError('{} is not a valid param for {}'.format(param_name, type(obj)))
        return obj

    def set_param(self, name, value):
        if hasattr(self, name):
            self.__setattr__(name, value)
        else:
            raise ValueError('{} is not a valid param for Config'.format(name))

    def summary(self):
        for attr in self.__dict__:
            value = self.__getattribute__(attr)
            print('{} : {}'.format(attr, value))

if __name__ == '__main__':
    config = Config()
    # config.summary()
    callbacks = config.learning_callbacks
    print(callbacks[0].patience)
    print(callbacks[0].monitor)

