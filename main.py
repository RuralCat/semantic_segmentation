import warnings
from model import Unet
from data_processing import load_images_train_data, image_data_generator
from config import Config, ImageConfig


if __name__ == '__main__':
    # ignore warnings
    warnings.filterwarnings("ignore")
    # prepare config
    config = ImageConfig()
    config.lr = 3e-4

    # create model
    model = Unet(config=config)

    # load data
    x, y = load_images_train_data(model, img_num=100)

    # date generator
    date_gen = image_data_generator()

    # run model
    model.run_model(x, y, use_generator=False, date_gen=date_gen)






