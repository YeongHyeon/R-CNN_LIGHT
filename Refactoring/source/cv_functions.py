import cv2

import numpy as np

def rgb2gray(rgb=None):

    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

def bluring(gray=None, k_size=11):

    return cv2.GaussianBlur(gray, (k_size, k_size), 0)

def adaptiveThresholding(gray=None, neighbor=5, blur=False, k_size=3):

    if(blur):
        gray = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neighbor, 1)

def erosion(binary_img=None, k_size=5, iterations=1):

    kernel = np.ones((k_size, k_size),np.uint8)

    return cv2.erode(binary_img, kernel, iterations=iterations)

def dilation(binary_img=None, k_size=5, iterations=1):

    kernel = np.ones((k_size, k_size),np.uint8)

    return cv2.dilate(binary_img, kernel, iterations=iterations)

def custom_opeing(binary_img=None, k_size=5, iterations=1):

    kernel = np.ones((k_size, k_size),np.uint8)
    tmp_ero = cv2.erode(binary_img, kernel, iterations=iterations)

    return cv2.dilate(tmp_ero, kernel, iterations=iterations)

def custom_closing(binary_img=None, k_size=5, iterations=1):

    kernel = np.ones((k_size, k_size),np.uint8)
    tmp_dil = cv2.dilate(binary_img, kernel, iterations=iterations)

    return cv2.erode(tmp_dil, kernel, iterations=iterations)

def opening(binary_img=None, k_size=2, iterations=1):

    kernel = np.ones((k_size, k_size), np.uint8)

    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations) # iteration = loop

def closing(binary_img=None, k_size=2, iterations=1):

    kernel = np.ones((k_size, k_size), np.uint8)

    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations) # iteration = loop

def contouring(closed=None):

    # return two values: contours, hierarchy
    return cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
