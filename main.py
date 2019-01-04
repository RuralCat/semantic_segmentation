import warnings
from model import Unet
from dataset import load_images_train_data, image_data_generator,\
    trainingset_augmentation
from config import Config, ImageConfig, ConfigOpt
import os

def augmentation(config):
    im_dir = os.path.join(config.root_path, 'data/golgi')
    mask_dir = os.path.join(config.root_path, 'data/golgi masks')
    output_dir = os.path.join(config.root_path, 'data/golgi images 0')
    gt_output_dir = os.path.join(config.root_path, 'data/golgi masks 0')
    trainingset_augmentation(im_dir, 512, 512,
                             samples=500, ground_truth_path=mask_dir,
                             output_dir=output_dir,
                             ground_truth_output_dir=gt_output_dir)


if __name__ == '__main__':
    # ignore warnings
    warnings.filterwarnings("ignore")

    # configuration
    config = ImageConfig()
    config.model_description = ''
    config.time = ''
    config.lr = 3e-4
    config.operation = ConfigOpt.TRAIN
    config.images_dir = '../data/golgi images 0'
    config.masks_dir = '../data/golgi masks 0'
    config.mean_map = 'mean_map_{}.pic'.format('')

    # prepare data (augmentation)
    if config.operation == ConfigOpt.AUGMENTATION:
        augmentation(config)

    if isinstance(config.operation, ConfigOpt) and \
            config.operation is not ConfigOpt.AUGMENTATION:
        # create model
        model = Unet(config=config)

        # load data
        x, y = load_images_train_data(model, img_num=8000)

        # date generator
        date_gen = image_data_generator()

        # run model
        model.run_model(x, y, use_generator=False, date_gen=date_gen)






