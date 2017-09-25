import numpy as np
import os, random, inspect

from tensorflow.contrib.learn.python.learn.datasets import base

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class DataSet(object):

    def __init__(self, who_am_i, class_len, data_len):

        self._who_am_i = who_am_i
        self._class_len = class_len
        self._data_len = data_len

    @property
    def amount(self):
        count = 0

        f = open(PACK_PATH+"/dataset/"+str(self._who_am_i)+".csv", 'r')
        while True:
            line = f.readline()
            if not line: break
            count = count + 1
        f.close()

        return count

    @property
    def class_num(self):
        return self._class_len

    @property
    def data_size(self):
        return self._data_len


    def next_batch(self, batch_size=10, start=-1, end=-1, nth=-1):
        audio_data = np.empty((0, self._data_len), float)
        audio_label = np.empty((0, self._class_len), int)

        with open(PACK_PATH+"/dataset/"+str(self._who_am_i)+".csv") as f:
            lines = f.readlines()

        if(nth == -1):
            if((start == -1) and (end == -1)):
                datas = random.sample(lines, batch_size)
            else:
                datas = lines[start:end]
        else:
            datas = []
            datas.append(lines[nth])

        for d in datas:
            data = d.split(',')
            label = data[0]
            audio = data[1:len(data)-1]

            audio = np.asarray(audio).reshape((1, len(audio)))

            audio_label = np.append(audio_label, np.eye(self._class_len)[int(np.asfarray(label))].reshape(1, self._class_len), axis=0)
            audio_data = np.append(audio_data, audio, axis=0)

        return audio_data, audio_label

def dataset_constructor():

    f = open(PACK_PATH+"/dataset/format.txt", 'r')
    class_len = int(f.readline())
    data_len = int(f.readline())
    f.close()

    train = DataSet(who_am_i="train", class_len=class_len, data_len=data_len)
    test = DataSet(who_am_i="test", class_len=class_len, data_len=data_len)
    valid = DataSet(who_am_i="valid", class_len=class_len, data_len=data_len)

    return base.Datasets(train=train, test=test, validation=valid)
