from datetime import datetime
import hashlib
import inspect, os
import random
import re
import struct
import sys
import shutil

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import cv2
import matplotlib.image as mpimg

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py"""

    """Builds a list of training images from the file system.
    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.
    Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.
    Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
    """
    print("\n***** Create image lists *****")

    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print(" Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print(' No files found')
            continue
        if len(file_list) < 20:
            print(' WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print(' WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                              (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

def key_from_dictionary(dictionary):
    print("\n***** Extract keys *****")
    master_key = list(dictionary.keys())
    sub_key = list(dictionary[master_key[0]].keys())

    print(" Master Key is...")
    print(" "+str(master_key))
    print(" Sub Key is...")
    print(" "+str(sub_key))

    return master_key, sub_key

def image_save(path, imagename, matrix):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path+imagename, matrix)

def imagelist_to_dataset(image_dir, image_lists, imsize=28):
    master_key, sub_key = key_from_dictionary(image_lists)

    print("\n***** Make image list *****")
    result_dir = "dataset/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)

    x_train = []
    t_train = np.empty((0), int)
    x_test = []
    t_test = np.empty((0), int)
    x_valid = []
    t_valid = np.empty((0), int)
    for key_i in [0, 1, 3]:
        if key_i == 0:
            result_name = "train"
        elif key_i == 1:
            result_name = "test"
        else:
            result_name = "valid"
        sys.stdout.write(" Make \'"+result_name+" list\'...")
        # m: class
        for m in master_key:

                for i in range(len(image_lists[m][sub_key[key_i]])):
                    # m: category
                    # image_lists[m][sub_key[key_i]][i]: image name
                    image_path = "./"+image_dir+"/"+m+"/"+image_lists[m][sub_key[key_i]][i]
                    # Read jpg images and resizing it.
                    origin_image = cv2.imread(image_path)
                    resized_image = cv2.resize(origin_image, (imsize, imsize))
                    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                    image_save(result_dir+"origin/"+result_name+"/", image_lists[m][sub_key[key_i]][i], origin_image)
                    image_save(result_dir+"resize/"+result_name+"/", image_lists[m][sub_key[key_i]][i], resized_image)
                    image_save(result_dir+"gray/"+result_name+"/", image_lists[m][sub_key[key_i]][i], grayscale_image)

                    if key_i == 0:
                        x_train.append(resized_image)
                        t_train = np.append(t_train, np.array([int(np.asfarray(m))]), axis=0)
                    elif key_i == 1:
                        x_test.append(resized_image)
                        t_test = np.append(t_test, np.array([int(np.asfarray(m))]), axis=0)
                    else:
                        x_valid.append(resized_image)
                        t_valid = np.append(t_valid, np.array([int(np.asfarray(m))]), axis=0)

        print(" complete.")
    #print(" x_train shape: " + str(np.array(x_train).shape))
    #print(" t_train shape: " + str(np.array(t_train).shape))
    #print(" x_test shape: " + str(np.array(x_test).shape))
    #print(" t_test shape: " + str(np.array(t_test).shape))
    x_train = np.asarray(x_train)
    t_train = np.asarray(t_train)
    x_test = np.asarray(x_test)
    t_test = np.asarray(t_test)
    return (x_train, t_train), (x_test, t_test), len(master_key)

def imagelist_to_tensor(image_dir, image_lists, imsize=28):
    X_data = []
    files = glob.glob ("*.jpg")
    for myFile in files:
        image = cv2.imread (myFile)
        X_data.append (image)

    print('X_data shape:', np.array(X_data).shape)

def dataset(image_dir, test_percentage, validation_percentage, imsize=28):
    test_percentage = 10
    validation_percentage = 10
    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, test_percentage, validation_percentage)

    (x_train, t_train), (x_test, t_test), classes = imagelist_to_dataset(image_dir=image_dir, image_lists=image_lists, imsize=imsize)
    print("\n Data set is ready!")
    print(" Data for train : " + str(x_train.shape[0]))
    print(" Data for test  : " + str(x_test.shape[0]))

    return (x_train, t_train), (x_test, t_test), classes

#=========================================
#              *** main ***
#=========================================
if __name__ == "__main__":
    dataset("images", 10, 10)
