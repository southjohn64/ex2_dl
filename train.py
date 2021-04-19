import os
import random
from tensorflow.keras.utils import plot_model
from siamese_net import SiameseNet
from utils.data_loader import split_train_val, load_dataset
import pathlib


def run_siamnet_train(dataset_zip_path: str, train_file_path , test_file_path, batch_size=2):
    random.seed(42)
    input_image_shape = (105, 105, 1)

    print('images zip path: {}'.format(dataset_zip_path))

    print('train file: {} , test file: {}'.format(train_file_path, test_file_path))

    train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list, \
    test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list = load_dataset(
        dataset_zip_path,
        train_file_path,
        test_file_path)
    print('train dataset has {} samples'.format(len(train_dataset[0])))
    print('test dataset has {} samples'.format(len(test_dataset[0])))

    train_dataset, y_array_train, val_dataset, y_array_val = split_train_val(train_dataset, y_array_train)
    print('train (after val split) dataset has {} samples'.format(len(train_dataset[0])))
    print('val dataset has {} samples'.format(len(val_dataset[0])))

    
    num_train_samples = len(train_dataset[0])
    
    curr_script_dir = pathlib.Path(__file__).parent.absolute()
    saved_model_dir = os.path.join(curr_script_dir, 'saved_models')
    if not os.path.isdir(saved_model_dir):
        os.mkdir(saved_model_dir)


    siamnet = SiameseNet(input_image_shape, num_train_samples, batch_size)
    model_img_file = saved_model_dir+'/siam_model_architecture.png'
    plot_model(siamnet.siam_model, to_file=model_img_file, show_shapes=True)


    siamnet.train(train_dataset, y_array_train, val_dataset, y_array_val, save_model_path=saved_model_dir,
                  batch_size=batch_size)

if __name__ == '__main__':
    run_siamnet_train(r'C:\Users\USER\Google Drive\Master_deg\למידה עמוקה\ex2\data_\lfwa.zip',
                      r'C:\Users\USER\Google Drive\Master_deg\למידה עמוקה\ex2\pairsDevTrain.txt',
                      r'C:\Users\USER\Google Drive\Master_deg\למידה עמוקה\ex2\pairsDevTest.txt',
                      batch_size=1)
