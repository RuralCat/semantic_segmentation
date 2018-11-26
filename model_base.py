from keras.layers import Conv2D, Input, MaxPooling2D, Dense, Flatten, Dropout, Cropping2D, Concatenate, Conv2DTranspose, BatchNormalization, Lambda, regularizers, initializers, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend import squeeze
import keras.backend as K
import os.path as op
import os
import data_processing as dp
import data_processing.augmentation as aug
from keras.utils import to_categorical
from PIL import Image
from tqdm import tqdm
import numpy as np


def TernaryModel(bdropout = True):
    # input
    input = Input(shape=(51, 51, 3, ))
    # conv 1
    conv1 = Conv2D(25, (4,4), activation='relu')(input)
    if bdropout: conv1 = Dropout(0.1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    # conv 2
    conv2 = Conv2D(50, (5,5), activation='relu')(conv1)
    if bdropout: conv2 = Dropout(0.2)(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    # conv 3
    conv3 = Conv2D(80, (6,6), activation='relu')(conv2)
    if bdropout: conv3 = Dropout(0.25)(conv3)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
    # dense
    flat = Flatten()(conv3)
    dense1 = Dense(1024, activation='relu')(flat)
    if bdropout: dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(1024, activation='relu')(dense1)
    if bdropout: dense2 = Dropout(0.5)(dense2)
    # output
    output = Dense(3, activation='softmax')(dense2)
    # create model
    model = Model(inputs=input, outputs=output)

    return model

class ModelBase:
    def __init__(self):
        pass

    def compile_model(self, lr=0.001):
        adam = Adam(lr=lr)
        self.model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def run_model(self, x, y, weights=None):
        # create callbacks
        callbacks = []
        # reduce learning rate when settled
        reduce_lr = ReduceLROnPlateau(monitor='loss',
                                      factor=0.4,
                                      patience=2,
                                      min_delta=1e-2,
                                      cooldown=0)
        callbacks.append(reduce_lr)
        # early stop
        if True:
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            callbacks.append(early_stop)
        # check point
        check_point = ModelCheckpoint(weights,
                                      monitor='val_loss',
                                      save_best_only=True,
                                      save_weights_only=True)
        callbacks.append(check_point)
        # tensorboard
        weights_name = op.basename(weights)
        log_dir = os.path.join('results', 'logs', weights_name)
        if not op.exists(log_dir):
            os.mkdir(log_dir)
        tensor_board = TensorBoard(log_dir=log_dir,
                                   histogram_freq=2,
                                   write_images=True,
                                   update_freq=1000)
        callbacks.append(tensor_board)
        # train
        self.model.fit(x, y,
                       batch_size = 4,
                       epochs=30,
                       verbose=1,
                       shuffle=True,
                       validation_split=0.05,
                       callbacks=callbacks)
        self.model.save_weights(weights)

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

def UpConvBlock0(input_layer0, input_layer1, crop_size, size, name=None):
    crop0 = Cropping2D((crop_size, crop_size))(input_layer0)
    upconv0 = Conv2DTranspose(size, (2,2), strides=(2,2))(input_layer1)
    cat0 = Concatenate()([crop0, upconv0])
    c0 = conv2d_bn(cat0, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    return c1

def Squeeze(x):
    return squeeze(x, axis=-1)

class Unet(ModelBase):
   def __init__(self):
       # image input
       img_input = Input(shape=(572, 572, 3, ))

       # channel size
       cs = 24

       # down sampling
       c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
       c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
       c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
       c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
       c5= ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

       # upsampleing
       upconv0 = UpConvBlock0(c4, c5, 4, 8 * cs, name='upconv_block0')
       upconv1 = UpConvBlock0(c3, upconv0, 16, 4 * cs, name='upconv_block1')
       upconv2 = UpConvBlock0(c2, upconv1, 40, 2 * cs, name='upconv_block2')
       upconv3 = UpConvBlock0(c1, upconv2, 88, cs, name='upconv_block3')

       # mask output
       mask_output = Conv2D(1, (1,1), activation='sigmoid', name='output')(upconv3)
       # mask_output = Lambda(Squeeze)(mask_output)

       self.model = Model(img_input, mask_output)

   @staticmethod
   def load_training_data(by_image=False, img_num=None):
       if by_image:
           # get images' path
           root_path = op.abspath('./')
           data_path = op.join(root_path, 'data_processing', 'normed images')
           mask_path = op.join(root_path, 'data_processing', 'mask')
           imgs_path = dp.get_allfile(data_path)
           masks_path = dp.get_allfile(mask_path)
           # read image and augmentation
           train_x = []
           train_y = []
           img_num = len(imgs_path) if img_num is None else min(img_num, len(imgs_path))
           for imp, mp, ind in zip(imgs_path, masks_path, range(img_num)):
               print('{}/{} Processing {}...'.format(ind + 1, img_num, op.basename(imp)))
               raw_image = dp.RawImage(imp, mp)
               imgs, masks = raw_image.augmentation()
               train_x.extend(imgs)
               train_y.extend(masks)
       else:
           # get path
           data_dir = op.join('data_processing', 'normed images', 'output')
           data_path = dp.get_allfile(data_dir)
           img_num = np.int32(len(data_path) / 2)
           imgs_path = data_path[:img_num]
           masks_path = data_path[img_num:]
           # read image
           train_x = np.zeros((img_num, 572, 572, 3), dtype=np.float32)
           train_y = np.zeros((img_num, 388, 388, 1), dtype=np.float32)
           with tqdm(total=img_num, desc='Processing', unit='Images') as p_bar:
               for imp, mp, ind in zip(imgs_path, masks_path, range(img_num)):
                   with Image.open(imp) as im:
                       train_x[ind] = np.array(im) / 255
                   with Image.open(mp) as mask:
                       train_y[ind][:,:,0] = aug.crop(np.array(mask) / 255, 92, 92, 388, 388)
                   p_bar.set_description('Processing {}'.format(op.basename(imp)))
                   p_bar.update(1)

        # normlize
       train_x -= np.mean(train_x, axis=0)

       # shuffle
       idx = np.random.permutation(train_x.shape[0])
       train_x = train_x[idx]
       train_y = train_y[idx]

       return train_x, train_y

class InceptionUnet(ModelBase):
    def __init__(self):
        pass

class Unetv2(ModelBase):
    def __init__(self):
        # image input
        img_input = Input(shape=(572, 572, 3,))

        # channel size
        cs = 24

        # down sampling
        c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
        c1_e = ConvBlock0(c1, cs, False, 'conv_block0_e')
        c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
        c2_e = ConvBlock0(c2, 2 * cs, False, 'conv_block1_e')
        c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
        c3_e = ConvBlock0(c3, 4 * cs, False, 'conv_block2_e')
        c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
        c4_e = ConvBlock0(c4, 8 * cs, False, 'conv_block3_e')
        c5 = ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

        # upsampleing
        upconv0 = UpConvBlock0(c4_e, c5, 2, 8 * cs, name='upconv_block0')
        upconv1 = UpConvBlock0(c3_e, upconv0, 14, 4 * cs, name='upconv_block1')
        upconv2 = UpConvBlock0(c2_e, upconv1, 38, 2 * cs, name='upconv_block2')
        upconv3 = UpConvBlock0(c1_e, upconv2, 86, cs, name='upconv_block3')

        # mask output
        mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='output')(upconv3)
        # mask_output = Lambda(Squeeze)(mask_output)

        self.model = Model(img_input, mask_output)

class TriangularNet(ModelBase):
    def __init__(self):
        # image input
        img_input = Input(shape=(572, 572, 3, ))
        cs = 24

        # down sampling
        c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
        c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
        c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
        c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
        c5 = ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

        # first upsampling
        fir_upconv0 = UpConvBlock0(c1, c2, 4, cs, name='first_upconv_block0')
        fir_upconv1 = UpConvBlock0(c2, c3, 4, 2 * cs, name='first_upconv_block1')
        fir_upconv2 = UpConvBlock0(c3, c4, 4, 4 * cs, name='first_upconv_block2')
        fir_upconv3 = UpConvBlock0(c4, c5, 4, 8 * cs, name='first_upconv_block3')

        # second upsampling
        sec_upconv0 = UpConvBlock0(fir_upconv0, fir_upconv1, 10, cs, name='second_upconv_block0')
        sec_upconv1 = UpConvBlock0(fir_upconv1, fir_upconv2, 10, 2 * cs, name='second_upconv_block1')
        sec_upconv2 = UpConvBlock0(fir_upconv2, fir_upconv3, 10, 4 * cs, name='second_upconv_block2')

        # third upsampling
        tir_upconv0 = UpConvBlock0(sec_upconv0, sec_upconv1, 22, cs, name='third_upconv_block0')
        tir_upconv1 = UpConvBlock0(sec_upconv1, sec_upconv2, 22, 2 * cs, name='third_upconv_block1')

        # final & output
        fou_upconv0 = UpConvBlock0(tir_upconv0, tir_upconv1, 46, cs, name='fou_upconv_block0')
        mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='output')(fou_upconv0)
        # mask_output = Lambda(Squeeze)(mask_output)

        self.model = Model(img_input, mask_output)


if __name__ == '__main__':
    unet_model = Unetv2()
    unet_model.model.summary()
    # tri_model = TriangularNet()
    # tri_model.model.summary()

