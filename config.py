# /usr/bin/python3.6

import keras.optimizers as opt
import keras.callbacks as kc
import os
import datetime

class Config(object):

    def __init__(self):
        # data path
        self.root_path = os.path.abspath('../')

        # model configurations
        self.model_name = 'unet'
        self.model_description = 'no1234skip'
        self.operation = 'train' # 'train', 'predict', 'evaluate'

        # learning configurations
        # optimizer
        self.opt = 'adam'
        # loss
        self.loss = 'binary_crossentropy'
        # metrics
        self.metrics = ['accuracy']
        # learning rate
        self.lr = 3e-4
        # momentum
        self.momentum = 0.95
        # batch size
        self.batch_size = 4
        # epochs
        self.epochs = 40
        # validation
        self.validation_split = 0.05

        # post-learning configurations
        # weights_path
        self.time = self._time
        self.weights_path = os.path.join(self.root_path, 'code/results', 'weights', self._weights)
        # log dir
        self.log_dir = os.path.join(self.root_path, 'code/results', 'logs', self._weights)
        # learning callbacks configurations
        self.use_earlystopping = True
        self.earlystopping_config = dict(monitor='val_loss',
                                         patience=50)
        self.use_reduceLR = True
        self.reduceLR_config = dict(monitor='loss',
                                    factor=0.4,
                                    patience=2,
                                    min_delta=1e-2,
                                    cooldown=0)
        self.use_checkpoint = True
        self.checkpoint_config = dict(monitor='val_loss',
                                      save_best_only=True,
                                      save_weights_only=True)
        self.use_tensorborad = False
        self.tensorborad_config = dict(histogram_freq=2,
                                       write_images=True,
                                       update_freq=1000)

    @property
    def optimizer(self):
        if self.opt == 'adam':
            return opt.Adam(self.lr)

    @property
    def learning_callbacks(self):
        callbacks = []
        # early stopping
        if self.use_earlystopping:
            callbacks.append(kc.EarlyStopping(**self.earlystopping_config))
        # reduce learning rate
        if self.use_reduceLR:
            callbacks.append(kc.ReduceLROnPlateau(**self.reduceLR_config))
        # check point
        if self.use_checkpoint:
            callbacks.append(kc.ModelCheckpoint(self.weights_path,
                                                **self.checkpoint_config))
        # tensorborad
        if self.use_tensorborad:
            callbacks.append(kc.TensorBoard(self.log_dir,
                                            **self.tensorborad_config))

        return callbacks

    @property
    def _weights(self):
        return 'model_{}_{}_{}.weights'.format(self.model_name, self.model_description, self.time)

    @property
    def _time(self):
        time_now = datetime.datetime.now()
        return '{}{}_{}{}'.format(time_now.month, time_now.day,
                                  time_now.hour, time_now.minute)

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

    def write_record_to_excel(self, res, **kwargs):
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = 'Test'
        # TO DO

class ImageConfig(Config):
    def __init__(self):
        Config.__init__(self)

        # data configuration
        self.images_dir = os.path.join(self.root_path, 'data\images')
        self.masks_dir = os.path.join(self.root_path, 'data\masks')
        self.mean_map = os.path.join(self.root_path, 'mean_map.pic')

if __name__ == '__main__':
    # config = ImageConfig()
    # config.summary()
    # print(os.path.exists(config.images_dir))
    a = {}
    b = dict(po=1, pl=2)
    print(type(a))
    print(type(b))


