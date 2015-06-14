import numpy as np
import math

def getLandmarksOfTooth(landmarks, toothNum):
    Outlandmarks = np.zeros((14,80))
    for i in range(14):
        Outlandmarks[i] = landmarks[i][toothNum][:]
    return Outlandmarks

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)) #calculate the distances of two landmarks in the sample

def calcMean(landmarks):
    '''calculates mean of landmarks. takes in 4d array of (14,8,80) and returns mean array of shape (8,80)'''
    mean = np.zeros([8,80])
    for i in range(landmarks.shape[1]):
        for k in range(landmarks.shape[2]):
            tot = 0
            for j in range(landmarks.shape[0]):
                tot += landmarks[j][i][k]
            mean[i][k] = tot/landmarks.shape[0]
    return mean

def calcMeanOneTooth(landmarks):
    '''calculates mean of landmarks of single tooth. takes in 2d array of (14,80) and returns mean array of shape (80)'''
    mean = np.zeros([80])
    for i in range(landmarks.shape[1]):
        tot = 0
        for j in range(landmarks.shape[0]):
            tot += landmarks[j][i]
        mean[i] = tot/landmarks.shape[0]
    return mean
    
def calcCenter(landmarks):
    '''calculates center of shape defined by landmarks of single tooth. takes in 1d array of (80) and returns x and y coordinate of center'''
    xCenter = 0
    yCenter = 0
    nbOfLandmarks = len(landmarks)/2
    for i in range(nbOfLandmarks):
        xCenter += landmarks[i*2]
        yCenter += landmarks[i*2+1]
    xCenter = xCenter/nbOfLandmarks
    yCenter = yCenter/nbOfLandmarks
    return xCenter, yCenter
    
def valuesConvergenceCheck(oldValues, newValues):
    for i in range(len(oldValues)):
        if abs(oldValues[i] - newValues[i]) > 10e-10:
            return False
    return True