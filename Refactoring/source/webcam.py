import os, sys, inspect, time
import cv2

import tensorflow as tf
import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

frame = None
height, width, chennel = None, None, None
content = None

def rgb2gray(rgb=None):

    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

def bluring(gray=None, k_size=11):

    return cv2.GaussianBlur(gray, (k_size, k_size), 0)

def adaptiveThresholding(gray=None, neighbor=5, blur=False, k_size=3):

    if(blur):
        gray = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neighbor, 1)

def opening(binary_img=None, k_size=2, iterations=1):

    kernel = np.ones((k_size, k_size), np.uint8)

    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1) # iteration = loop

def closing(binary_img=None, k_size=2, iterations=1):

    kernel = np.ones((k_size, k_size), np.uint8)

    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1) # iteration = loop

def contouring(closed=None):

    # return two values: contours, hierarchy
    return cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def draw_boxes(boxes=None):

    global frame

    for b in boxes:
        x, y, w, h, result, acc = b

        txt_color=(100, 100, 100)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),1)
        cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

def load_format():

    global height, width, chennel, content

    f = open(PACK_PATH+"/dataset/format.txt", 'r')
    class_len = int(f.readline())
    data_len = int(f.readline())
    height = int(f.readline())
    width = int(f.readline())
    chennel = int(f.readline())
    f.close()

    f = open(PACK_PATH+"/dataset/labels.txt", 'r')
    content = f.readlines()
    f.close()
    for idx in range(len(content)):
        content[idx] = content[idx][:len(content[idx])-1] # rid \n

def img2predict(image=None):

    global height, width, chennel

    resized_image = cv2.resize(image, (height, width))

    return np.asarray(resized_image).reshape((1, height*width*chennel))

def prediction(origin=None, contours=None, sess=None, training=None, prediction=None, saver=None):

    global content

    if(os.path.exists(PACK_PATH+"/checkpoint/checker.index")):
        saver.restore(sess, PACK_PATH+"/checkpoint/checker")

        boxes = []
        pad = 15

        for cnt in contours:

            area = cv2.contourArea(cnt)
            if((area < 50) or (area > 2500)):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            x, y, w, h = x-pad, y-pad, w+pad, h+pad

            if((x > 0) and (y > 0)):

                if((x < origin.shape[1]) and (y < origin.shape[0])): # check: box in the region

                    prob = sess.run(prediction, feed_dict={x:img2predict(image=origin[y:y+h, x:x+w]), training:False})
                    result = str(content[int(np.argmax(prob))])
                    acc = max(prob)

                    if(acc > 0.85):
                        boxes.append([x, y, w, h, result, acc])

    boxes = sorted(boxes, key=lambda l:l[5], reverse=True)

    return boxes

def webcam_main(sess=None, x_holder=None, training=None, prediction=None, saver=None):

    global frame, content

    load_format()

    if(os.path.exists(PACK_PATH+"/checkpoint/checker.index")):
        saver.restore(sess, PACK_PATH+"/checkpoint/checker")

        camera = cv2.VideoCapture(0)

        cv2.namedWindow("frame")

        while True:

            (grabbed, frame) = camera.read()

            frame = cv2.flip(frame,1)

            if not grabbed:
                break

            gray = rgb2gray(rgb=frame)
            # cv2.imshow("gray", gray)

            binary_img = adaptiveThresholding(gray=gray, neighbor=5, blur=True, k_size=7)
            # cv2.imshow("binary_img", binary_img)

            opened = opening(binary_img=binary_img, k_size=2, iterations=1)
            # cv2.imshow("opened", opened)

            closed = closing(binary_img=opened, k_size=4, iterations=3)
            # cv2.imshow("closed", closed)

            contours, _ = contouring(closed=closed)

            boxes = []
            pad = 15
            std_time = time.time()
            classification_counter = 0
            for cnt in contours:

                area = cv2.contourArea(cnt)
                if((area < 50) or (area > 2500)):
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                x, y, w, h = x-pad, y-pad, w+pad, h+pad

                if((x > 0) and (y > 0)):

                    if((x < frame.shape[1]) and (y < frame.shape[0])): # check: box in the region

                        prob = sess.run(prediction, feed_dict={x_holder:img2predict(image=frame[y:y+h, x:x+w]), training:False})
                        classification_counter += 1

                        result = str(content[int(np.argmax(prob))])
                        acc = np.max(prob)

                        if(acc > 0.85):
                            boxes.append([x, y, w, h, result, acc])

            sys.stdout.write(' %.3f [classify/sec]\r' %(classification_counter/(time.time() - std_time)))
            sys.stdout.flush()

            boxes = sorted(boxes, key=lambda l:l[5], reverse=True) # sort by acc

            draw_boxes(boxes=boxes)

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
