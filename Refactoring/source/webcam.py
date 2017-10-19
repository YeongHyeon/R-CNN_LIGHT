import os, sys, inspect, time
import cv2

import tensorflow as tf
import numpy as np

import cv_functions

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

frame = None
height, width, chennel = None, None, None
content = None

def draw_boxes(boxes=None):

    global frame

    for b in boxes:
        x, y, w, h = b

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),1)

def draw_predict_boxes(boxes=None):

    global frame

    eye = 0
    for b in boxes:
        x, y, w, h, result, acc = b

        txt_color = (100, 100, 100)
        if((result == "open") or (result == "close")):
            eye += 1
            # if(eye > 2):
                # break
            if(result == "open"):
                txt_color = (255, 0, 0)
            elif(result == "close"):
                txt_color = (0, 0, 255)

            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),1)
            cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)
        else:
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),1)
            # cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            # cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)
            cv2.putText(frame, "Others "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(frame, "Others "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

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

def img_predict(image=None):

    global height, width, chennel

    resized_image = cv2.resize(image, (height, width))

    return np.asarray(resized_image).reshape((1, height*width*chennel))

def region_predict(origin=None, contours=None, sess=None, x_holder=None, training=None, prediction=None, saver=None):

    global content

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

                prob = sess.run(prediction, feed_dict={x_holder:img_predict(image=frame[y:y+h, x:x+w]), training:False})
                classification_counter += 1

                result = str(content[int(np.argmax(prob))])
                acc = np.max(prob)

                if(acc > 0.85):
                    boxes.append([x, y, w, h, result, acc])

    sys.stdout.write('%.3f [classify/sec]\r' %(classification_counter/(time.time() - std_time)))
    sys.stdout.flush()

    boxes = sorted(boxes, key=lambda l:l[4], reverse=True) # sort by result
    boxes = sorted(boxes, key=lambda l:l[5], reverse=True) # sort by acc
    return boxes

def webcam_main(sess=None, x_holder=None, training=None, prediction=None, saver=None):

    global frame

    print("")

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

            gray = cv_functions.rgb2gray(rgb=frame)
            # cv2.imshow("gray", gray)

            binary_img = cv_functions.adaptiveThresholding(gray=gray, neighbor=5, blur=True, k_size=7)
            # cv2.imshow("binary_img", binary_img)

            opened = cv_functions.opening(binary_img=binary_img, k_size=2, iterations=1)
            # cv2.imshow("opened", opened)

            closed = cv_functions.closing(binary_img=opened, k_size=4, iterations=3)
            # cv2.imshow("closed", closed)

            cus_opened = cv_functions.custom_opeing(binary_img=binary_img, ero_size=3, dil_size=7, iterations=1)
            # cv2.imshow("cus_opened", cus_opened)

            cus_closed = cv_functions.custom_closing(binary_img=binary_img, ero_size=3, dil_size=5, iterations=1)
            # cv2.imshow("cus_closed", cus_closed)

            contours, _ = cv_functions.contouring(binary_img=cus_opened)

            boxes = cv_functions.contour2box(contours=contours, padding=15)
            # draw_boxes(boxes=boxes)

            boxes_pred = region_predict(origin=frame, contours=contours, sess=sess, x_holder=x_holder, training=training, prediction=prediction, saver=saver)

            draw_predict_boxes(boxes=boxes_pred)

            cv2.imshow("frame", frame)

            key = cv2.waitKey(1) & 0xFF
            # press 'p' to Pause
            if(key == ord("p")):
                cv2.waitKey(0)
            # press 'q' to Quit
            elif(key == ord("q")):
                print("\n\nQUIT")
                break

        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
