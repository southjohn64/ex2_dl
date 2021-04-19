import os
from tqdm import tqdm
import csv
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
import numpy as np
import cv2


def load_data(file_path):
	with open(file_path, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter='\t'))[1:]
	return rows


def load_images(dataset_zip_path, rows):
	'''

	:param images_folder:
	:param rows:
	:return: mp arrays of images,y_true, titles
	'''
	
	path = Path(dataset_zip_path)
	parent_dir = path.parent.absolute()
	

	first_image_list = []
	second_image_list = []

	first_image_title_list = []
	second_image_title_list = []

	y_true_array = []
	
	num_of_images = len(rows)

	for i, row in tqdm(iterable=enumerate(rows), total=num_of_images):
		first_image, second_image, y_true, first_image_title, second_image_title = loadImagePairFromRow(
			dataset_zip_path,
			row)
		first_image_list.append(first_image)
		second_image_list.append(second_image)

		y_true_array.append(y_true)

		first_image_title_list.append(first_image_title)
		second_image_title_list.append(second_image_title)

	first_image_list = np.array(first_image_list).astype('float32')
	second_image_list = np.array(second_image_list).astype('float32')
	y_true_array = np.array(y_true_array)
	first_image_title_list = np.array(first_image_title_list)
	second_image_title_list = np.array(second_image_title_list)
	return first_image_list, second_image_list, y_true_array, first_image_title_list, second_image_title_list


def loadImagePairFromRow(dataset_zip_path, row):
	image_title = []
	if len(row) == 3:
		# same
		person_name = row[0]

		first_image_number = row[1]
		second_image_number = row[2]

		first_image = loadImage(dataset_zip_path, person_name, first_image_number)
		first_image_title = person_name + "_" + first_image_number

		second_image = loadImage(dataset_zip_path, person_name, second_image_number)
		second_image_title = person_name + "_" + second_image_number

		return first_image, second_image, 1.0, first_image_title, second_image_title
	else:
		# different
		person_name = row[0]
		first_image_number = row[1]

		second_person_name = row[2]
		second_image_number = row[3]

		first_image_title = person_name + "_" + first_image_number
		second_image_title = second_person_name + "_" + second_image_number

		first_image = loadImage(dataset_zip_path, person_name, first_image_number)
		second_image = loadImage(dataset_zip_path, second_person_name, second_image_number)

		return first_image, second_image, 0.0, first_image_title, second_image_title


def loadImage(images_folder, person_name, image_number):
	filename = r"{0}/{1}/{1}_{2:04d}.jpg".format(images_folder, person_name, int(image_number))
	im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

	#im = Image.open(filename).convert('L')
	im = resize(im)
	im = np.expand_dims(np.array(im), -1)
	return im  # (250,250,1)


def resize(im):
	dim = (105,105)
	resized = cv2.resize(im ,dim, interpolation = cv2.INTER_AREA)
	return resized


def split_train_datasets(train_dataset, ratio=0.1):
	num_train_samples = int(len(train_dataset) * (1.0 - ratio))

	train = train_dataset[:num_train_samples]
	val = train_dataset[num_train_samples:]
	return train, val


def print_images(row, row_title):
	fig, axes = plt.subplots(nrows=1, ncols=2)
	axis = axes.ravel()
	axis[0].imshow(row[0])
	axis[0].set_title(row_title[0])
	axis[1].imshow(row[1])
	axis[1].set_title(row_title[1])
	plt.show()


def save_dataset_to_npy(data_dir, train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list,test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list):
	print('saving dataset to npy files')
	is_npy_saved = False
	try:

	# check for npy file
		train_npy = os.path.join(data_dir , 'pairsDevTrain.npy')
		np.save(train_npy,train_dataset)

		y_array_train_npy = os.path.join(data_dir , 'y_array_train.npy')
		np.save(y_array_train_npy,y_array_train)

		first_image_title_train_list_npy = os.path.join(data_dir , 'first_image_title_train_list.npy')
		np.save(first_image_title_train_list_npy,first_image_title_train_list)

		second_image_title_train_list_npy = os.path.join(data_dir , 'second_image_title_train_list.npy')
		np.save(second_image_title_train_list_npy,second_image_title_train_list)

		test_npy = os.path.join(data_dir , 'pairsDevTest.npy')
		np.save(test_npy,test_dataset)

		y_array_test_npy = os.path.join(data_dir , 'y_array_test.npy')
		np.save(y_array_test_npy,y_array_test)

		first_image_title_test_list_npy = os.path.join(data_dir , 'first_image_title_test_list.npy')
		np.save(first_image_title_test_list_npy,first_image_title_test_list)

		second_image_title_test_list_npy = os.path.join(data_dir , 'second_image_title_test_list.npy')
		np.save(second_image_title_test_list_npy,second_image_title_test_list)
		is_npy_saved = True
	
	except Exception as e:
		is_npy_saved = False
		print(e)
		raise
	print('saved dataset to npy files in: ' , data_dir)
	return is_npy_saved

def load_dataset_from_npy(data_dir):
	print('loading dataset from npy files in: ' , data_dir)
	is_npy_loaded = False
	
	train_dataset = None
	y_array_train = None
	first_image_title_train_list = None
	second_image_title_train_list = None
	
	test_dataset= None
	y_array_test = None
	first_image_title_test_list = None
	second_image_title_test_list = None
	
	try:

	# check for npy file
		train_npy = os.path.join(data_dir , 'pairsDevTrain.npy')
		if Path(train_npy).is_file():
			train_dataset = np.load(train_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False
		
		y_array_train_npy = os.path.join(data_dir , 'y_array_train.npy')
		if Path(y_array_train_npy).is_file():
			y_array_train = np.load(y_array_train_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False

		first_image_title_train_list_npy = os.path.join(data_dir , 'first_image_title_train_list.npy')
		if Path(first_image_title_train_list_npy).is_file():
			first_image_title_train_list = np.load(first_image_title_train_list_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False

		second_image_title_train_list_npy = os.path.join(data_dir , 'second_image_title_train_list.npy')
		if Path(second_image_title_train_list_npy).is_file():
			second_image_title_train_list = np.load(second_image_title_train_list_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False
	
		test_npy = os.path.join(data_dir , 'pairsDevTest.npy')
		if Path(test_npy).is_file():
			test_dataset = np.load(test_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False
	
		y_array_test_npy = os.path.join(data_dir , 'y_array_test.npy')
		if Path(y_array_test_npy).is_file():
			y_array_test = np.load(y_array_test_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False

		first_image_title_test_list_npy = os.path.join(data_dir , 'first_image_title_test_list.npy')
		if Path(first_image_title_test_list_npy).is_file():
			first_image_title_test_list = np.load(first_image_title_test_list_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False

		second_image_title_test_list_npy = os.path.join(data_dir , 'second_image_title_test_list.npy')
		if Path(second_image_title_test_list_npy).is_file():
			second_image_title_test_list = np.load(second_image_title_test_list_npy)
			is_npy_loaded = True
		else:
			is_npy_loaded = False
		
	except Exception as e:
		is_npy_loaded = False
		print(e)
		raise
	if is_npy_loaded:
		print('loaded dataset from npy files in: ' , data_dir)
	else:
		print('no npy files found')
	  
	return is_npy_loaded ,train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list, \
		test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list

def load_dataset(dataset_zip_path, train_file, test_file):
	'''

	:param images_folder:
	:param train_file:
	:param test_file:
	:return:
	'''

	
	path = Path(dataset_zip_path)
	data_dir = path.parent.absolute()
	
	is_npy_loaded ,train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list, \
		test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list = load_dataset_from_npy(data_dir)

	if not is_npy_loaded:
	  # check if the zip was extracted already

	  images_folder = os.path.join(data_dir, 'lfw2/lfw2')

	  if not os.path.isdir(images_folder):
		  with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
			  zip_ref.extractall(data_dir)

	  train_rows = load_data(train_file)
	  first_image_train_list, second_image_train_list, y_array_train, first_image_title_train_list, second_image_title_train_list = load_images(
		  images_folder, train_rows)

	  # normalize data
	  first_image_train_list = pre_process(first_image_train_list)
	  second_image_train_list = pre_process(second_image_train_list)

	  train_dataset = [first_image_train_list, second_image_train_list]

	  test_rows = load_data(test_file)
	  first_image_test_list, second_image_test_list, y_array_test, first_image_title_test_list, second_image_title_test_list = load_images(
		  images_folder,
		  test_rows)

	  # normalize data
	  first_image_test_list = pre_process(first_image_test_list)
	  second_image_test_list = pre_process(second_image_test_list)
	  test_dataset = [first_image_test_list, second_image_test_list]

	  save_dataset_to_npy(data_dir, train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list,test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list)

	return train_dataset, y_array_train, first_image_title_train_list, second_image_title_train_list, \
		   test_dataset, y_array_test, first_image_title_test_list, second_image_title_test_list


def pre_process(image_list):
	return image_list / 255


def split_train_val(train_dataset, y_array_train, ratio=0.1):
	train_ratio = 1.0 - ratio

	total_samples = len(train_dataset[0])
	train_samples = int(total_samples * train_ratio)

	val_dataset = [train_dataset[0][train_samples:], train_dataset[1][train_samples:]]
	y_array_val = y_array_train[train_samples:]

	train_dataset = [train_dataset[0][:train_samples], train_dataset[1][:train_samples]]
	y_array_train = y_array_train[:train_samples]

	return train_dataset, y_array_train, val_dataset, y_array_val

# if __name__ == '__main__':
#	 images_folder = r'C:\Users\USER\Desktop\lfwa\lfw2\lfw2'
#	 train_file = r'C:\Users\USER\Desktop\lfwa\lfw2\lfw2\pairsDevTrain.txt'
#	 test_file = r'C:\Users\USER\Desktop\lfwa\lfw2\lfw2\pairsDevTest.txt'
#	 train_dataset, y_array_train, train_titles, test_dataset, y_array_test, test_titles = load_dataset(images_folder,
#																										train_file,
#																										test_file)
#	 print_images(train_dataset[0], train_titles[0])
