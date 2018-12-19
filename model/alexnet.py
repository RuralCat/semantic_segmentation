
from model.modelbase import *

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