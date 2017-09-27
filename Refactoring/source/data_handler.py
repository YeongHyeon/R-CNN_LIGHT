import os, glob, random, inspect, shutil

import cv2
import numpy as np

# custom modules
import utility as utility
import constructor as constructor

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def split_data(path=None, directories=None, extensions=None, clone=0):

    print("\n** Split whole datas")

    if(not(os.path.exists(path))):
        print("Path not exists \"" + str(path) + "\"")
        return

    for di in directories:
        if(not(os.path.exists(PACK_PATH+"/train/"+di))):
            os.mkdir(PACK_PATH+"/train/"+di)
        if(not(os.path.exists(PACK_PATH+"/test/"+di))):
            os.mkdir(PACK_PATH+"/test/"+di)
        if(not(os.path.exists(PACK_PATH+"/valid/"+di))):
            os.mkdir(PACK_PATH+"/valid/"+di)

    for di in directories:
        fi_list = utility.get_filelist(directory=path+"/"+di, extensions=extensions)

        fi_list = random.sample(fi_list, len(fi_list))

        tr_point = int(len(fi_list)*0.8)
        te_point = int(len(fi_list)*0.9)
        va_point = int(len(fi_list)*1.0)

        train = fi_list[:tr_point]
        test = fi_list[tr_point:te_point]
        valid = fi_list[te_point:va_point]

        utility.copy_file(train, PACK_PATH+"/train/"+di, clone=clone)
        utility.copy_file(test, PACK_PATH+"/test/"+di, clone=clone)
        utility.copy_file(valid, PACK_PATH+"/valid/"+di, clone=clone)

    print("Split the datas!")

def make_dataset(category=None, dirlist=None, height=32, width=32, extensions=None):

    print("\n** Make "+category+".csv")

    class_len = len(dirlist)
    io_mode = "w"
    label_number = 0

    for di in dirlist:
        fi_list = utility.get_filelist(directory=PACK_PATH+"/"+category+"/"+di, extensions=extensions)

        for fi in fi_list:

            image = cv2.imread(fi)
            resized_image = cv2.resize(image, (height, width))
            height, width, chennel = resized_image.shape
            resized_image = resized_image.reshape((height*width*chennel))

            utility.save_dataset_to_csv(save_as=category, label=label_number, data=resized_image, mode=io_mode)
            io_mode = "a"

        label_number += 1

    if(os.path.exists(PACK_PATH+"/"+category)): # management storage
        shutil.rmtree(PACK_PATH+"/"+category)

    f = open(PACK_PATH+"/dataset/format.txt", "w")
    f.write(str(label_number))
    f.write("\n")
    f.write(str(height*width*chennel))
    f.write("\n")
    f.write(str(height))
    f.write("\n")
    f.write(str(width))
    f.write("\n")
    f.write(str(chennel))
    f.close()

def check():

    print("\n** Check dataset")

    check_path = PACK_PATH+"/dataset"
    print(check_path)
    in_data = 0
    if(utility.check_directory(dir_name=check_path)):
        for fi in glob.glob(check_path+"/*"):
            in_data += 1

    if(in_data >= 5):
        return True
    else:
        return False

def make(path=None, height=32, width=32, extensions=None, clone=0):

    print("\n** Make dataset")

    check_list = ["dataset", "train", "test", "valid"]
    cate_list = ["train", "test", "valid"]
    shuffle_list = ["train", "test"]

    for ch in check_list:
        utility.refresh_directory(PACK_PATH+"/"+ch)

    dirlist = utility.get_dirlist(path=path)
    split_data(path=path, directories=dirlist, extensions=extensions, clone=clone)

    for di in dirlist:
        fi_list = utility.get_filelist(directory=path+"/"+di, extensions=extensions)
    print("I got the standard shape!")

    for ca in cate_list:
        make_dataset(category=ca, dirlist=dirlist, height=height, width=width, extensions=extensions)

    for shu in shuffle_list:
        utility.shuffle_csv(filename=PACK_PATH+"/dataset/"+shu)

def load():

    print("\n** Load dataset")

    dataset = constructor.dataset_constructor()

    print("Num of Train datas : "+str(dataset.train.amount))
    print("Num of Test  datas : "+str(dataset.test.amount))
    print("Num of Valid datas : "+str(dataset.validation.amount))

    return dataset
