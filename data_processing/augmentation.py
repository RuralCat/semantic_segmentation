
import cv2
import numpy as np
import collections
import bisect
from numba import jit

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

@jit
def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst