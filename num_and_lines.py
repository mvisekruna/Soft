import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import keras
from keras.models import load_model
import math


def deskew(img): 
    m = cv2.moments(img)

    if abs(m['mu02']) < 1e-2:
        return img
    skew = m['mu11']/m['mu02']
    M = np.array([[1, skew, -0.5*28*skew], [0, 1, 0]], 'float32')
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def blueLine(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowBlue = np.array([110, 50, 50])
    uppBlue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lowBlue, uppBlue)
    edges = cv2.Canny(mask, 125, 150)


    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap = 50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2, = line[0]
            cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
    return (x1, y1), (x2, y2)
    #cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    #cv2.imshow("edges", edges)

def greenLine(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 15, 15])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap = 10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2  = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return (x1, y1), (x2, y2)
    #cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    #cv2.imshow("edges", edges)

def blueRect(frame, coordsBlue):

    (x1, y1), (x2, y2) = coordsBlue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1,1), np.uint8) 
    
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width = thresh.shape

    for contour, hier in zip(contours, hierarchy):
        center, size, angle = cv2.minAreaRect(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        width, height = size
        if width > 1 and width < 30 and height > 1 and height < 30:
            if width > 9 or height > 9:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                A = (x, y)
                B = (x + w, y)
                C = (x, y + h)
                D = (x + w, y + h)

                rect = (A,B,C,D)
                             
                k = (y1 - y2)/(x1 - x2) 
                kA = (y-y1)/(x-x1)
                kB = (y-y1)/((x+w)-x1)
                kC = ((y+h)-y1)/(x-x1)
                kD = ((y+h)-y1)/((x+w)-x1)
                            
                myArray = []
                myArray.append(kA)
                myArray.append(kB)
                myArray.append(kC)
                myArray.append(kD)
                            
                nearest = min(myArray, key=lambda x:abs(x-k)) #formula za izracunavanje najblizeg broja 
                            
                if k<=nearest<=k+0.045:
                    if nearest == kA:
                        (i, j) = A
                    elif nearest == kB:
                        (i, j) = B
                    elif nearest == kC:
                        (i, j) = C
                    else:
                        (i, j) = D

                    if (x1<=i<=x2 and y2<=j<=y1):
                        t = (2*x+w)/2
                        s = (2*y+h)/2
                        return t, s

def greenRect(frame, coordsGreen):

    (x1, y1), (x2, y2) = coordsGreen
    #(x,y,w,h) = rect

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1,1), np.uint8) 
    
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width = thresh.shape

    for contour, hier in zip(contours, hierarchy):
        center, size, angle = cv2.minAreaRect(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        width, height = size
        if width > 1 and width < 30 and height > 1 and height < 30:
            if width > 9 or height > 9:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                A = (x, y)
                B = (x + w, y)
                C = (x, y + h)
                D = (x + w, y + h)

                rect = (A,B,C,D)
                 
                k = (y1 - y2)/(x1 - x2) 
                            
                kA = (y-y1)/(x-x1)
                kB = (y-y1)/((x+w)-x1)
                kC = ((y+h)-y1)/(x-x1)
                kD = ((y+h)-y1)/((x+w)-x1)
                            
                myArray = []
                myArray.append(kA)
                myArray.append(kB)
                myArray.append(kC)
                myArray.append(kD)
                            
                nearest = min(myArray, key=lambda x:abs(x-k)) #formula za izracunavanje najblizeg broja 
                            
                if k<=nearest<=k+0.045:
                    if nearest == kA:
                        (i, j) = A
                    elif nearest == kB:
                        (i, j) = B
                    elif nearest == kC:
                        (i, j) = C
                    else:
                        (i, j) = D

                    if (x1<=i<=x2 and y2<=j<=y1):
                        t = (2*x+w)/2
                        s = (2*y+h)/2
                        return t, s
   
def cropImgPredictNum(frame, point, model):
    x,y = point
    x,y = int(x), int(y)
    cropImg = frame[y-12:y+12, x-12:x+12]
    cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
    cropImg = deskew(cropImg)
    cropImg = cv2.dilate(cropImg, (4, 4))

    toPredict = cropImg.flatten() / 255.0
    toPredict = (np.array([toPredict], 'float32'))
    modelNum = np.argmax(model.predict(toPredict)[0])

    return modelNum






