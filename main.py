import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import keras
from keras.models import load_model
import math
import num_and_lines as nl

finalRes = 0
suma = 0
razlika = 0
sumArray = [] 
sumArray.append(0)
subArray = []
subArray.append(0)

def getSuma():
    global suma
    return suma

def getRazlika():
    global razlika
    return razlika

def getSumArray(): 
    global sumArray
    return sumArray

def getSubArray(): 
    global subArray
    return subArray

def sumElem(number): 
    global sumArray
    sumArray.append(number)

def subElem(number): 
    global subArray
    subArray.append(number)
    
def sumOperation(number): 
    global suma
    suma+=number

def subOperation(number): 
    global razlika
    razlika+=number

def finishSum(number):
    sumArray = getSumArray()
    for i in range (len(sumArray)):
        if(i == len(sumArray)-1):
            temp = sumArray[i]
            if(temp != number):
                sumElem(number)
                sumOperation(number)
                #print("Saberi: " + str(number))
            else:
                sumOperation(0)

def finishSub(number):
    subArray = getSubArray()
    for i in range (len(subArray)):
        if(i == len(subArray)-1):
            temp = subArray[i]
            if(temp != number):
                subElem(number)
                subOperation(number)
                #print("Saberi: " + str(number))
            else:
                subOperation(0)

model = load_model('net2.h5')

f = open('out.txt', 'w') # otvaram out za pisanje

outStr = 'RA 175/2015 Milica Visekruna\n'
outStr += 'file\tsum\n'

firstPart = 'videos/video-'
lastPart = '.avi'

for i in range(0, 10):
    title = firstPart + str(i) + lastPart
    video = cv2.VideoCapture(title)

    foundRects = []
    foundNums = []

    while(video.isOpened()): 
        ret, frame = video.read()
            
        if not ret:
            video = cv2.VideoCapture(title)
            continue
            
        lineB = nl.blueLine(frame) #koordinate plave linije
        bluePoint = nl.blueRect(frame, lineB)
        if(bluePoint!=None):
            num1 = nl.cropImgPredictNum(frame, bluePoint, model)
            finishSum(num1)

        lineG = nl.greenLine(frame) #koordinate zelene linije
        greenPoint = nl.greenRect(frame, lineG)
        if(greenPoint!=None):
            num2 = nl.cropImgPredictNum(frame, greenPoint, model)
            finishSub(num2)

        
        cv2.imshow('frame', frame)
                #da bi mogli da zatvorimo prozor
        key = cv2.waitKey(1) #30 sec after each frame
        if key == 27: #esc je 27
            break

    video.release()
    cv2.destroyAllWindows()

    resultSum = getSuma()
    resultSub = getRazlika()

    finalRes = resultSum - resultSub
    print('Video: ' + title)
    print('  Ukupno: ' + str(finalRes))
    print('-----------------')

    outStr += title + ' ' + str(finalRes) + '\n'

    finalRes = 0
    suma = 0
    razlika = 0
    sumArray = [] 
    sumArray.append(0)
    subArray = []
    subArray.append(0)

f.write(outStr)
f.close()

        