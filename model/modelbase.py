from keras.layers import Conv2D, Input, MaxPooling2D, Dense, Flatten, Dropout, Cropping2D, Concatenate, Conv2DTranspose, BatchNormalization, Lambda, regularizers, initializers, Activation
import keras.layers as KL
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from config import Config


class ModelBase:
    def __init__(self, config=None):
        if config is not None:
            assert isinstance(config, Config)
            self.config = config
        else:
            self.config = Config()

    def _reset_param(self, param, value):
        if value is not None:
            self.config.set_param(param, value)

    def compile_model(self, lr=None):
        self._reset_param('lr', lr)
        self.model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def run_model(self, x, y, weights=None, use_genetaror=False, **kwargs):
        # reset weights
        self._reset_param('weights_name', weights)
        # create callbacks
        callbacks = self.config.learning_callbacks
        # train
        assert isinstance(self.model, Model)
        if use_genetaror:
            data_gen = kwargs.get('date_gen', None)
            assert isinstance(data_gen, ImageDataGenerator)
            date_gen_flow = data_gen.flow(x, y, batch_size=self.config.batch_size)
            self.model.fit_generator(date_gen_flow,
                                     epochs=self.config.epochs,
                                     steps_per_epoch=len(x) / self.config.batch_size,
                                     verbose=1,
                                     callbacks=callbacks)
        else:
            self.model.fit(x, y,
                           batch_size=self.config.batch_size,
                           epochs=self.config.epochs,
                           verbose=1,
                           shuffle=True,
                           validation_split=self.config.validation_split,
                           callbacks=callbacks)
        self.model.save_weights(self.config.weights_name)

    def predict(self, weights, x):
        # load weights
        self.model.load_weights(weights)
        # predict
        y = self.model.predict(x)

        return y

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


if __name__ == '__main__':
    # x,y = Unet.load_training_data()
    pass

