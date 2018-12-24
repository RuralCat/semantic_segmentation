import warnings
from model import Unet
from data_processing import load_images_train_data, image_data_generator,\
    trainingset_augmentation
from config import Config, ImageConfig
import os

def augmentation(config):
    im_dir = os.path.join(config.root_path, 'data/images')
    mask_dir = os.path.join(config.root_path, 'data/masks')
    output_dir = os.path.join(config.root_path, 'data/images_0')
    gt_output_dir = os.path.join(config.root_path, 'data/masks_0')
    trainingset_augmentation(im_dir, 512, 512,
                             samples=8000, ground_truth_path=mask_dir,
                             output_dir=output_dir,
                             ground_truth_output_dir=gt_output_dir)


if __name__ == '__main__':
    # ignore warnings
    warnings.filterwarnings("ignore")

    # prepare config
    config = ImageConfig()
    config.lr = 3e-4

    # prepare data (augmentation)
    if False:
        augmentation(config)

    if config.operation in ['train', 'predict', 'evaluate']:
        # create model
        model = Unet(config=config)

        # load data
        x, y = load_images_train_data(model, img_num=8000)

        # date generator
        date_gen = image_data_generator()

        # run model
        model.run_model(x, y, use_generator=False, date_gen=date_gen)






