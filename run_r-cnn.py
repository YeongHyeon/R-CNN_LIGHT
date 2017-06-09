# coding: utf-8
import sys, os, inspect, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append(os.pardir)
import math
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import dataset
from custom_convnet import ConvNet # import convolution neural network class
from common.trainer import Trainer # import trainer class
from common.functions import *
from common.util import shuffle_dataset
import cv2
from scipy.spatial import distance as dist
from datetime import datetime
import time
"""
This project is designed to detect user's blinking in the video.
If you want to fix it, you can do it.(You want to detect something else ...)
It should be noted that the complexity of CNN has been reduced for real-time processing.
So please understand if you have a little lack of accuracy.

Although tensorflow api is required in this project, tensorflow is not used at all in training.
Tensorflow only uses images to distinguish them for testing validation for training, and also reveals the source of the function.
Tensorflow used in the 'dataset_loader' module.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
"""

"""
'PACK_PATH' is a variable that tells you where your package is located.
Please use it well.
For example, when saving an image ...
"""
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

network = None
imsize = 32

classes = 0 # I recommend that you do not modify the dataset because it is automatically updated during the configuration process.
"""
The first data I provide is features of the face such as eyes, nose, and mouth.
If you are reorganizing the data, please modify the strings in this list. (The 'class_name')
"""
class_name = ['Close', 'Open', 'Eyebrow', 'Nose', 'Mouth', 'Hair', 'Background']

frame = None
std_time = 0

def chenneling(x):
    """
    This function makes the dataset suitable for training.
    Especially, gray scale image does not have channel information.
    This function forces one channel to be created for gray scale images.
    """

    # if grayscale image
    if(len(x.shape) == 3):
        C = 1
        N, H, W = x.shape
        x = np.asarray(x).reshape((N, H, W, C))
    else: # color image
        pass

    x = x.transpose(0, 3, 1, 2)

    x = x.astype(float)

    return x

def frame_predict(origin_image, network, imsize=28):
    """
    It takes an image similar to each class and returns the most similar class value.
    """

    resized_image = cv2.resize(origin_image, (imsize, imsize))

    if(len(resized_image.shape) == 2):
        N = 1
        C = 1
        H, W = resized_image.shape
        resized_image = np.asarray(resized_image).reshape((N, H, W, C))
    else:
        N = 1
        H, W, C = resized_image.shape
        resized_image = np.asarray(resized_image).reshape((N, H, W, C))

    resized_image = resized_image.transpose(0, 3, 1, 2)
    resized_image = resized_image.astype(float)

    img_predict = network.predict(resized_image)
    predict_sm = softmax(img_predict)

    predict_list = []
    for idx in range(len(predict_sm[0])):
        predict_list.append(predict_sm[0][idx])

    max_class = predict_list.index(max(predict_list))
    return max_class, max(predict_list)

# Image save Function
def image_save(label=None, matrix=None):
    """
    You can use it to check if the image is well classified.
    You will be hard categorized in the place where you saved the package, so check it in.
    """

    now = datetime.now()
    filename = now.strftime('%Y%m%d_%H%M%S%f')+".jpg"
    directory = PACK_PATH+'/frames/'
    if(label!=None):
        directory += '/'+str(label)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory+filename, matrix)
    return directory+filename

def classification_by_contour(origin, imsize=28):
    """
    Referenced by http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    """

    global frame, network, std_time
    kernel, closing = None, None
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1)

    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    pad = 15
    std_time = time.time()
    pre_cnt = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area < 50) or (area > 2500)):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x-pad, y-pad, w+pad, h+pad
        if((x > 0) and (y > 0)):
            if((x < origin.shape[1]) and (y < origin.shape[0])):
                if(w > h*1.2):
                    result, acc = frame_predict(origin[y:y+h, x:x+w], network, imsize=imsize)
                    pre_cnt += 1
                    if(acc > 0.80):
                        """Try annotating when you want to check if the classification is working."""
                        #image_save(label=result, matrix=origin[y:y+h, x:x+w])
                        boxes.append([x, y, w, h, result, acc])

    #print(" %.3f [classify/sec]" %(pre_cnt/(time.time() - std_time)))
    sys.stdout.write(' %.3f [classify/sec]\r' %(pre_cnt/(time.time() - std_time)))
    sys.stdout.flush()

    txt_color=(0, 0, 0)
    eye_cnt = 0
    boxes = sorted(boxes, key=lambda l:l[5], reverse=True)
    for b in boxes:
        x, y, w, h, result, acc = b
        #cv2.rectangle(frame, (x,y-15), (x+w,y), (255, 255, 255), cv2.cv.CV_FILLED)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),1)

        if(not((result == 0) or ((result == 1)))):
            txt_color=(0, 0, 0)
            cv2.putText(frame, class_name[result]+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(frame, class_name[result]+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

        elif((result == 0) or ((result == 1))):

            if(result == 0):
                txt_color=(0, 0, 255)
            elif(result == 1):
                txt_color=(255, 0, 0)
            cv2.putText(frame, class_name[result]+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(frame, class_name[result]+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

            eye_cnt += 1
            #if(eye_cnt > 2):
            #    break



    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Closing", closing)


def cnn_constructor():
    """
    Referenced by https://github.com/oreilly-japan/deep-learning-from-scratch
    common modules referenced there too.
    """

    global network, classes, imsize

    (x_train, t_train), (x_test, t_test), classes = dataset(image_dir="images", test_percentage=10, validation_percentage=10, imsize=imsize)

    x_train = chenneling(x_train)
    x_test = chenneling(x_test)

    train_num = x_train.shape[0]
    test_num = x_test.shape[0]

    x_train, t_train = shuffle_dataset(x_train, t_train)
    x_test, t_test = shuffle_dataset(x_test, t_test)

    net_param = "cnn_params"+str(imsize)+".pkl"
    if not os.path.exists("params/"):
        os.makedirs("params/")

    # make convolution eural network
    # x_train.shape[1:] returns channel, height, width
    network = ConvNet(input_dim=(x_train.shape[1:]),
                            conv_param = {'filter_num': 20, 'filter_size': 3, 'pad': 0, 'stride': 1},
                            hidden_size=32, output_size=classes, weight_init_std=0.001)

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=1, mini_batch_size=FLAGS.batch_size,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=train_num)

    params_loaded = False
    if not os.path.exists("params/"):
        os.makedirs("params/")
    if(os.path.exists("params/"+net_param)):
        network.load_params("params/"+net_param)
        params_loaded = True
        print("\n* Loaded Network Parameters!  -  "+net_param)
    if((FLAGS.train_epochs > 0) or (params_loaded == False)):
        if(FLAGS.train_epochs <= 0):
            FLAGS.train_epochs = 10
        # Training
        for ep in range(FLAGS.train_epochs):
            trainer.train()
            # Save parameters
            network.save_params("params/"+net_param)
        print("\n* Saved Network Parameters!  -  "+net_param)

        # plot graphs
        # Grpah 1: Accuracy
        markers = {'train': 'o', 'test': 's', 'loss': 'd'}
        x1 = np.arange(FLAGS.train_epochs)
        plt.clf()
        plt.plot(x1, trainer.train_acc_list, marker='o', label='train', markevery=1)
        plt.plot(x1, trainer.test_acc_list, marker='s', label='test', markevery=1)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.1)
        plt.legend(loc='lower right')
        plt.title("Accuracy")
        now = datetime.now()
        filename = now.strftime('%Y%m%d_%H%M%S%f')+".png"
        plt.savefig(filename)
        #plt.show()

        # Graph 2: Loss
        x2 = np.arange(len(trainer.train_loss_list))
        plt.clf()
        plt.plot(x2, trainer.train_loss_list, marker='o', label='loss', markevery=1)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.legend(loc='lower right')
        plt.title("Cross entropy loss")
        now = datetime.now()
        filename = now.strftime('%Y%m%d_%H%M%S%f')+".png"
        plt.savefig(filename)
        #plt.show()

def main(source=0):
    global frame

    width_half = 25
    height_half = 25

    camera = cv2.VideoCapture(source)

    cv2.namedWindow("frame")

    while True:

        (grabbed, frame) = camera.read()

        if(source == 0):
            frame = cv2.flip(frame,1)

        if not grabbed:
            break

        classification_by_contour(frame, imsize=imsize)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        # press 'p' to Pause
        if(key == ord("p")):
            cv2.waitKey(0)
        # press 'q' to Quit
        elif(key == ord("q")):
    		print("\nQUIT")
    		break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=0, help='Default: 0. If you enter a value greater than 0, training proceeds.')
    parser.add_argument('--batch_size', type=int, default=10, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--source', type=str, default="", help='Default is webcam streaming. You can enter the path of the vedio for run it.')
    FLAGS, unparsed = parser.parse_known_args()

    if(len(FLAGS.source) <= 0):
        FLAGS.source = 0

    cnn_constructor()
    main(source=FLAGS.source)
