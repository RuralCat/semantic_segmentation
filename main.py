

import os
import warnings
from model_base import Unet, TriangularNet, Unetv2
from data_processing.dataset import get_allfile

if __name__ == '__main__':
    # ignore warnings
    warnings.filterwarnings("ignore")
    # prepare data_processing
    path = os.path.join(os.path.abspath('./'), 'data_processing', 'normed images')
    file_lists = get_allfile(path)
    # create model
    # model = TernaryModel()
    model = Unetv2()
    # model = TriangularNet()
    # complie model
    model.compile_model(1e-3)
    # load data
    x, y = Unet.load_training_data(by_image=False)
    # run model
    # weights_name = 'results/weights/model_1916_subtract_mean.weights'
    # weights_name = 'results/weights/model_1516_subtract_mean.weights'
    weights_name = 'results/weights/model_unetv2_1108_1820_cs24.weights'
    if True:
        model.run_model(x, y, weights_name)
    else:
        y = model.predict(weights_name, file_lists[0])
        mask = create_image_from_label(file_lists[0], y)


# 1101_1001 cs = 24 train loss = 0.10
# 1102 2008 cs = 32 train loss = 0.11 val_loss = 0.14
# 1104 2400 cs = 24 with original image, train_loss = 0.1474 val_loss = 0.3097






