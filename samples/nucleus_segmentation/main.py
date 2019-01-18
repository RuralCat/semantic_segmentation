import cv2
from dataset.dataset import get_all_file, load_images_train_data, \
    image_data_generator, load_image_test_data
import os
from model import Unet
from samples.nucleus_segmentation import convert_verts_to_mask
from config import ImageConfig, ConfigOpt
import numpy as np
from history import History as hist

def generate_gray_images():
    root_path = 'K:\\BIGCAT\\Projects\\Nuclei segmentation\\data\\Tissue images'
    save_path = os.path.join(root_path, 'Gray images')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_list = get_all_file(root_path)
    for file in file_list:
        im = cv2.imread(file)
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        p = os.path.join(save_path, os.path.basename(file)[:-4] + '.jpg')
        cv2.imwrite(p, gray_im)

def generate_mask():
    root_path = 'K:\\BIGCAT\\Projects\\Nuclei segmentation\\data\\Annotations'
    save_path = 'K:\\BIGCAT\\Projects\\Nuclei segmentation\\data\\Tissue images\\Mask images'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    image_shape = (1000, 1000)
    file_list = get_all_file(root_path)
    for file in file_list:
        p = os.path.join(save_path, os.path.basename(file)[:-4] + '.jpg')
        mask = convert_verts_to_mask(image_shape, file)
        cv2.imwrite(p, mask)

def rename():
    save_path = 'K:\\BIGCAT\\Projects\\Nuclei segmentation\\data\\Tissue images\\Gray images'
    file_list = get_all_file(save_path)
    for file in file_list:
        os.rename(file, file[:-4] + '.jpg')

if __name__ == '__main__':
    data_path = 'K:\\BIGCAT\\Projects\\Nuclei segmentation\\data\\Tissue images'
    root_path = os.path.abspath('../../../')

    # configuration
    model_description = 'nucleus_cs24_aug_inputbatch'
    config = ImageConfig(root_path, model_description)
    config.lr = 3e-4
    config.operation = ConfigOpt.PREDICT
    config.images_dir = os.path.join(data_path, 'Augmented gray images')
    config.masks_dir = os.path.join(data_path, 'Augmented mask images')
    config.mean_map = os.path.join(root_path, 'code/results/mean_map',
                                   'mean_map_{}.pic'.format(config.time))

    # prepare data

    # run
    if config.operation is not ConfigOpt.AUGMENTATION:
        # create model
        model = Unet(config=config)
        # load data
        if config.operation is ConfigOpt.TRAIN:
            x, y = load_images_train_data(model)
            # data generator
            data_gen = image_data_generator()
            # run model
            model.run_model(x, y, use_generator=False, data_gen=data_gen)
        elif config.operation is ConfigOpt.PREDICT:
            x = load_image_test_data(model, config.images_dir, 100)
            y_gt = load_image_test_data(model, config.masks_dir, 100)
            # data generator
            data_gen = image_data_generator()
            # run model
            config.time = '117_229'
            weights = os.path.join(config.root_path, 'code/results/weights', config._weights)
            y = model.run_model(x, weights=weights)
            print(y.shape)
            for i in range(y.shape[0]):
                cv2.imshow('predicted image',
                           np.concatenate((x[i][:,:,0], y[i][:,:,0], y_gt[i][:,:,0]), axis=1))
                cv2.waitKey()
