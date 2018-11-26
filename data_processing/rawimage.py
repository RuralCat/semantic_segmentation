
import skimage.io as skio
# from nucleus_visualize import plot_image
from data_processing.dataset import get_allfile, read_image, read_nucleus_bound, get_all_mask
import numpy as np
from data_processing import augmentation as aug
import Augmentor

# path
import os
ROOTPATH = os.path.abspath('../')
DATAPATH = os.path.join(ROOTPATH, 'data_processing', 'normed images')
ANOTPATH = os.path.join(ROOTPATH, 'data_processing', 'Annotations')

class Image:
    pass


class RawImage(Image):

    def __init__(self, data, mask_path=None):
        # read image data_processing
        if isinstance(data, str):
            self.image = read_image(data)
        else:
            self.image = data
        # read image mask
        self.mask = read_image(mask_path) if mask_path else None

    def plot(self):
        # plot_image(self.image)
        pass

    def augmentation(self):
        imgs = self._augmentation([self.image])
        masks = self._augmentation([self.mask])
        # plot images
        # for data_processing in imgs:
        #     plot_image(data_processing)
        return imgs, masks

    def _augmentation(self, data_stack):
        # rotate 45, 90, 135
        data_rot_stack = []
        for i in [45, 135]:
            d0= aug.augment_mask(data_stack, aug.rotate, angle=i)
            data_rot_stack.extend(d0)
        data_stack.extend(data_rot_stack)

        # flip: vertical flip, horizontal flip
        data_flip_stack = []
        for i in range(2):
            d0 = aug.augment_mask(data_stack, aug.flip, axis=i)
            data_flip_stack.extend(d0)
        # data_stack.extend(data_flip_stack)

        # crop at different scale & resize to destination size (like 572 X 572)
        data_crop_stack = []
        scales = [0.2, 0.3, 0.4]
        nums = [30, 30, 20]
        for scale, num in zip(scales, nums):
            sz = np.int32(np.floor(np.array(self.image.shape) * scale))
            d0 = aug.augment_mask(data_stack, aug.dense_crop,
                                  height=sz[0], width=sz[1], num=num)
            d0 = aug.augment_mask(d0, aug.resize, dst_size=(572, 572))
            data_crop_stack.extend(d0)
        data_stack = data_crop_stack

        return data_stack

    def convert_verts_to_mask(self, anotation_path, save_path=None):
        # read verts
        verts = read_nucleus_bound(anotation_path)
        bound_masks, inside_masks, _ = get_all_mask(verts, self.image.shape)
        # create mask map
        mask = np.zeros(self.image.shape[0:2], dtype='uint8')
        bound = np.zeros(self.image.shape[0:2], dtype='uint8')
        for in_mask in inside_masks:
            mask = mask | in_mask
            # bound += in_maks
        mask *= 255
        if save_path:
            skio.imsave(save_path, mask)
        # plot_image(mask)
        # plot_image(bound)

    @property
    def is_single_channel(self):
        return len(self.image.shape) == 1


def trainingset_augmentation(data_path, output_width, output_height,
                             samples=100,
                             ground_truth_path=None,
                             output_dir='output'):
    # create pipeline
    p = Augmentor.Pipeline(data_path, output_dir)
    if ground_truth_path: p.ground_truth(ground_truth_path)

    # add color operation
    # p.random_contrast(probability=1, min_factor=1, max_factor=1)
    # p.random_brightness(probability=1, min_factor=1, max_factor=1)
    # p.random_color(probability=1, min_factor=1, max_factor=1)

    # add shape operation
    p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
    p.crop_random(probability=1, percentage_area=0.3, randomise_percentage_area=False)
    p.random_distortion(probability=0.8, grid_width=30, grid_height=30, magnitude=1)
    p.skew(probability=0.5)
    p.shear(probability=0.4, max_shear_left=5, max_shear_right=5)
    p.resize(probability=1, width=output_width, height=output_height)

    # generate
    p.sample(samples)

if __name__ == '__main__':
    # test
    imgs_path = get_allfile(DATAPATH)
    anots_path = get_allfile(ANOTPATH)
    masks_path = get_allfile(os.path.join(ROOTPATH, 'data_processing', 'mask'))
    trainingset_augmentation(os.path.join(ROOTPATH, 'data_processing', 'normed images'),
                             os.path.join(ROOTPATH, 'data_processing', 'mask'))
    from PIL import Image
    im = Image.open('')
    im.close()

