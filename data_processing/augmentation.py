
import cv2
import numpy as np
import collections

class Augmentation:
    pass

def augment_mask(data_list, func, **kwargs):
    data_stack = []
    # should be list
    if not isinstance(data_list, list):
        data_list = [data_list]
    # augmentation
    for data in data_list:
        res = func(data, **kwargs)
        if isinstance(res, list):
            data_stack.extend(res)
        else:
            data_stack.append(res)

    return data_stack

# diffrent image augmentation method: crop, resize, flip, rotate ...
def crop(img, x, y, height, width):
    return img[x : x + height, y : y + width]


def dense_crop(img, height, width, num=20):
    sz = img.shape
    h_step = np.int32(np.floor((sz[0] - height) / num))
    w_step = np.int32(np.floor((sz[1] - width) / num))
    # crop
    data_stack = []
    for i in range(num):
        data_stack.append(crop(img, i * h_step, i * w_step, height, width))

    return data_stack

def resize(im, dst_size):
    return cv2.resize(im, dst_size)

def flip(im, axis=0):
    return cv2.flip(im, flipCode=axis)

def rotate(img, angle):
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img

def rot90(img):
    return np.rot90(img, k=1)

def rot180(img):
    return np.rot90(img, k=2)