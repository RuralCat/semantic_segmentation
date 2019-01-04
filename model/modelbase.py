from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from config import Config, ConfigOpt

class ModelBase:
    def __init__(self, model, config=None):
        # model
        assert isinstance(model, Model)
        self.model = model
        # config
        if config is not None:
            assert isinstance(config, Config)
            self.config = config
        else:
            self.config = Config()
        # set placeholder
        self.x = None
        self.y = None

    def _compile(self):
        self.model.compile(optimizer=self.config.optimizer,
                           loss=self.config.loss,
                           metrics=self.config.metrics)

    def _reset_param(self, param, value):
        if value is not None:
            self.config.set_param(param, value)

    def _fit(self, callbacks):
        self.model.fit(self.x, self.y,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       verbose=1,
                       shuffle=True,
                       validation_split=self.config.validation_split,
                       callbacks=callbacks)

    def _fit_generator(self, data_gen, callbacks):
        assert isinstance(data_gen, ImageDataGenerator)
        date_gen_flow = data_gen.flow(self.x, self.y, batch_size=self.config.batch_size)
        self.model.fit_generator(date_gen_flow,
                                 epochs=self.config.epochs,
                                 steps_per_epoch=self.num_samples / self.config.batch_size,
                                 verbose=1,
                                 callbacks=callbacks)

    def _dump_data(self, x, y=None, **kwargs):
        self.x = x
        self.y = y
        self.num_samples = x.shape[0]

    def _train(self, use_generator=False, **kwargs):
        # create callbacks
        callbacks = self.config.learning_callbacks
        # train
        assert isinstance(self.model, Model)
        if use_generator:
            data_gen = kwargs.get('date_gen', None)
            self._fit_generator(data_gen, callbacks)
        else:
            self._fit(callbacks)
        # save weights
        self.model.save_weights(self.config.weights_path)

    def _predict(self):
        # load weights
        self.model.load_weights(self.config.weights_path)
        # predict
        y = self.model.predict(self.x)
        return y

    def run_model(self, x, y=None, lr=None, weights=None, use_generator=False, **kwargs):
        # dump data
        self._dump_data(x, y, **kwargs)

        # reset paras
        self._reset_param('lr', lr)
        self._reset_param('weights_path', weights)

        # compile model
        self._compile()

        # run model
        op = self.config.operation
        assert isinstance(op, ConfigOpt)
        if op == ConfigOpt.TRAIN:
            self._train(use_generator, **kwargs)
        elif op == ConfigOpt.PREDICT:
             return self._predict()


def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False, name=''):
    """
    Utility function to apply conv + BN.
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    # padding = 'same'
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Conv2D(nb_filter,
               (num_row, num_col),
               name=name,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_regularizer=regularizers.l2(0.00004),
               kernel_initializer=initializers.VarianceScaling(scale=2.0,
                                                               mode='fan_in',
                                                               distribution='normal',
                                                               seed=None))(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x

def ConvBlock0(input_layer, size, pooling=True, name=''):
    c0 = conv2d_bn(input_layer, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    if pooling:
        p0 = MaxPooling2D(pool_size=(2,2))(c1)
        return c1, p0
    else:
        return c1

def Squeeze(x):
    return K.squeeze(x, axis=-1)

def UpConvBlock0(input_layer0, input_layer1, crop_size, size, name=None):
    crop0 = Cropping2D((crop_size, crop_size))(input_layer0)
    upconv0 = Conv2DTranspose(size, (2,2), strides=(2,2))(input_layer1)
    # upconv0 = KL.UpSampling2D((2,2), interpolation='nearest')(input_layer1)
    cat0 = Concatenate()([crop0, upconv0])
    c0 = conv2d_bn(cat0, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    return c1

def UpConvBlock1(layer, size, name=None):
    upconv0 = Conv2DTranspose(size, (2, 2), strides=(2, 2))(layer)
    c0 = conv2d_bn(upconv0, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    return c1

def UpConvBlock2(layer, size, name=''):
    upconv0 = Conv2DTranspose(size, (2, 2), strides=(2, 2))(layer)
    c0 = conv2d_bn(upconv0, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    c2 = MaxPooling2D(pool_size=(2,2))(c1)

    return c2



if __name__ == '__main__':
    # x,y = Unet.load_training_data()
    pass

