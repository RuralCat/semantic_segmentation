import warnings
from model import Unet
from data_processing import  get_allfile
from config import Config


if __name__ == '__main__':
    # ignore warnings
    warnings.filterwarnings("ignore")
    # prepare data_processing
    path = '..\data\images\output'
    file_lists = get_allfile(path)

    config = Config()

    # create model
    # model = TernaryModel()
    model = Unet()
    # model = TriangularNet()
    # complie model
    model.compile_model(1e-3)
    # load data
    # x, y = load_training_data(by_image=True, img_num=1000)
    x, y = None, None
    # run model
    weights_name = 'results/weights/model_unet_no1234skip_1218_1604_cs24.weights'
    if True:
        model.run_model(x, y, weights_name)
    else:
        y = model.predict(weights_name, file_lists[0])
        mask = create_image_from_label(file_lists[0], y)






