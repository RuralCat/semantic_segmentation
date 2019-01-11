
from dataset import read_image, augmentation
from scipy import ndimage
import numpy as np

class Metrics(object):
    def __init__(self, shape_constrains=False):
        """
        :param shape_constrains: if ground truth should have same shape with predicted data
        """
        self.shape_constrains = shape_constrains

    def _parse_data(self, gt, pd):
        """
        gt: ground truth,
        pd: predicted data
        """
        # read data
        if isinstance(gt, str):
            gt = read_image(gt)
        if isinstance(pd, str):
            pd = read_image(pd)
        # uniting data shape
        if gt.shape != pd.shape:
            if self.shape_constrains:
                raise ValueError("the shape of ground truth is not consitent with predicted's")
            else:
                pd = augmentation.resize(pd, gt.shape)
        return gt, pd


def rand_score():
    pass

class F1_Score(Metrics):
    def __init__(self):
        Metrics.__init__(self)

    def __call__(self, gt, pd, tau, *args, **kwargs):
        gt, pd = self._parse_data(gt, pd)

def find_boundingbox(mask):
    # label
    label_im, nb_labels = ndimage.label(mask)
    s = ndimage.find_objects(label_im, nb_labels)
    print(s[0])


def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

# test function
root_path = 'K:\BIGCAT\Projects\\'
def find_bb():
    im_path = 'EM\data\Other data\EM CA1 hippocampus region of brain/training_groundtruth.tif'
    im_path = root_path + im_path
    mask = read_image(im_path)
    find_boundingbox(mask[0])

    return mask

if __name__ == '__main__':
    mask = find_bb()
    mask0 = mask[0]
    print(mask0.shape)

