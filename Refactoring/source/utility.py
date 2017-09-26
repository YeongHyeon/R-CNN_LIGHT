import os, sys, glob, shutil, psutil, inspect, random
import scipy.misc

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def check_directory(dir_name):
    if(os.path.exists(dir_name)):
        return True
    else:
        return False

def check_file(file_path):
    if(os.path.isfile(file_path)):
        return True
    else:
        return False

def check_memory():
    pid = os.getpid()
    proc = psutil.Process(pid)
    used_mem = proc.memory_info()[0]

    print("Memory Used: %.2f GB\t( %.2f MB )" %(used_mem/(2**30), used_mem/(2**20)))

    return used_mem

def refresh_directory(dir_name):
    if(os.path.exists(dir_name)):
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    else:
        os.mkdir(dir_name)

def get_dirlist(path=None): # make directory list from path

    directories = []
    for dirname in os.listdir(path):
        directories.append(dirname)

    directories.sort()

    f = open(PACK_PATH+"/dataset/labels.txt", "w")
    for di in directories:
        f.write(str(di))
        f.write("\n")
    f.close()

    return directories

def get_filelist(directory=None, extentions=None): # make directory list from directory with path

    file_list = []
    for ext in extentions:
        for fi in glob.glob(directory+"/*."+ext):
            file_list.append(fi)

    print(str(len(file_list))+" files in "+directory)

    return file_list

def copy_file(origin, copy):
    count = 0
    for ori in origin:
        if(clone == 0):
            shutil.copy(ori, copy+"/"+str(count)+".jpg")
        else:
            for c in range(clone):
                shutil.copy(ori, copy+"/"+str(count)+"_clone_"+str(c)+".jpg")
        count = count + 1

def shuffle_csv(filename=None):
    f = open(filename+".csv", "r")
    lines = f.readlines()
    f.close()

    random.shuffle(lines)

    f = open(filename+".csv", "w")
    f.writelines(lines)
    f.close()

def save_dataset_to_csv(save_as="sample", label=None, data=None, mode='w'):

    f = open(PACK_PATH+"/dataset/"+save_as+".csv", mode)

    f.write(str(label))
    f.write(",")

    for da in data:
        f.write(str(da))
        f.write(",")

    f.write("\n")

    f.close()

def save_graph_as_image(train_list, test_list, ylabel=""):

    print(" Save "+ylabel+" graph in ./graph")

    x = np.arange(len(train_list))
    plt.clf()
    plt.plot(x, train_list, label="train "+ylabel)
    plt.plot(x, test_list, label="test "+ylabel, linestyle='--')
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.ylim(-0.1, max([1, max(train_list), max(test_list)])*1.1)
    if(ylabel == "accuracy"):
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    #plt.show()

    if(not(os.path.exists("./graph"))):
        os.mkdir("./graph")
    else:
        pass
    now = datetime.now()

    plt.savefig("./graph/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+ylabel+".png")

def save_confusion(save_as="sample", labels=None, lists=None, size=None):

    print(" Save confusion in ./confusion")

    confusion = np.empty((0, size), float)

    for li in lists:
        tmp_confu = li[0]

        for idx in range(li.shape[0]):
            if(idx == 0):
                pass
            else:
                tmp_confu = np.sum((tmp_confu, li[idx]), axis=0) # sum the same label probs

        tmp_confu = tmp_confu / li.shape[0] # divide total prob

        confusion = np.append(confusion, np.asarray(tmp_confu).reshape((1, len(tmp_confu))), axis=0)

    if(not(check_directory(PACK_PATH+"/confusion"))):
        os.mkdir(PACK_PATH+"/confusion")

    result = np.kron(confusion, np.ones((confusion.shape[0]*100, confusion.shape[1]*100))) # pump the matrix for save image

    now = datetime.now()

    # save as csv
    f = open(PACK_PATH+"/confusion/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+save_as+".csv", "w")

    f.write("X")
    f.write(",")
    for la in labels:
        f.write(la)
        f.write(",")
    f.write("Classification overall")
    f.write(",")
    f.write("Producer Accuracy (Precision)")
    f.write(",")
    f.write("\n")

    for row in range(confusion.shape[0]):
        f.write(labels[row])
        f.write(",")

        overall_cla = np.sum(confusion[row])
        precision = confusion[row][row] / overall_cla
        for con_elm in confusion[row]:
            f.write(str(round(con_elm, 5)))
            f.write(",")
        f.write(str(round(overall_cla, 5)))
        f.write(",")
        f.write(str(round(precision, 5)))
        f.write(",")
        f.write("\n")

    confusion_t = np.transpose(confusion)

    f.write("Truth overall")
    f.write(",")
    for row in range(confusion_t.shape[0]):
        overall_tru = confusion_t[row].shape[0]
        f.write(str(round(overall_cla, 5)))
        f.write(",")
    f.write("\n")

    f.write("User Accuracy (Recall)")
    f.write(",")
    for row in range(confusion_t.shape[0]):
        overall_tru = np.sum(confusion_t[row])
        recall = confusion[row][row] / confusion_t[row].shape[0]
        f.write(str(round(precision, 5)))
        f.write(",")
    f.write("\n")

    f.close()

    result[0][0] = 1
    # save as image
    scipy.misc.imsave(PACK_PATH+"/confusion/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+save_as+".jpg", result)
