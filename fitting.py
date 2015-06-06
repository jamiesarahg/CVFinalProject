import cv2
import numpy as np
import math

def calculateXYGradients(images, show=False):
    xGradientImages = []
    yGradientImages = []
    for img in images:
        #sobelx = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)))
        sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
        xGradientImages.append(sobelx)
        #sobely = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)))
        sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
        yGradientImages.append(sobely)
        if show:
            cv2.imshow('original',cv2.resize(img, (0,0), fx=0.25, fy=0.25))
            cv2.waitKey(0)
            cv2.imshow('xGradient',cv2.resize(sobelx, (0,0), fx=0.25, fy=0.25))
            cv2.waitKey(0)
            cv2.imshow('yGradient',cv2.resize(sobely, (0,0), fx=0.25, fy=0.25))
            cv2.waitKey(0)
    return xGradientImages, yGradientImages

def calculateLandmarkNormals(shapeLandmarks):
    #shapeLandmarks is an  array containing interleaved x and y coordinates for all landmarks of one shape
    normals = []
    range = len(shapeLandmarks)/2
    for i in range(range):
        centerX = shapeLandmarks[2*i]
        centerY = shapeLandmarks[2*i+1]
        if(i==0):
            prevX = shapeLandmarks[2*(range-1)]
            prevY = shapeLandmarks[2*(range-1)+1]
        else:
            prevX = shapeLandmarks[2*(i-1)]
            prevY = shapeLandmarks[2*(i-1)+1]
        if(i==range-1):
            nextX = shapeLandmarks[0]
            nextY = shapeLandmarks[1]
        else:
            nextX = shapeLandmarks[2*(i+1)]
            nextY = shapeLandmarks[2*(i+1)+1]
    normalX, normalY = calculateLandmarkNormal(prevX, prevY, centerX, centerY, nextX, nextY)
    normals.append(normalX, normalY)
    return normals
    
def calculateLandmarkNormal(prevX, prevY, centerX, centerY, nextX, nextY):
    #calculate the first surface vector
    prevVectorX = prevX-centerX
    prevVectorY = prevY-centerY
    #calculate the first surface normal
    n1x, n1y = calculatePerpendicularVector(prevVectorX, prevVectorY)
    #make sure this surface normal is pointing outwards
    if(crossProductPositive(prevVectorX, prevVectorY, n1x, n1y) is True):
        n1x = -n1x
        n1y = -n1y
    #normalize this surface normal
    n1length = lengthOfVector(n1x, n1y)
    n1x = n1x / n1length
    n1y = n1y / n1length
    #calculate the second surface vector
    nextVectorX = nextX-centerX
    nextVectorY = nextY-centerY
    #calculate the second surface normal
    n2x, n2y = calculatePerpendicularVector(nextVectorX, nextVectorY)
    #make sure this surface normal is pointing outwards
    if(crossProductPositive(nextVectorX, nextVectorY, n2x, n2y) is False):
        n2x = -n2x
        n2y = -n2y
    #normalize this surface normal
    n2length = lengthOfVector(n2x, n2y)
    n2x = n2x / n2length
    n2y = n2y / n2length
    #the required landmark normal is just the sum of the two calculated surface normals
    normalX = n1x + n2x
    normalY = n1y + n2y
    normalLength = lengthOfVector(normalX, normalY)
    normalX = normalX / normalLength
    normalY = normalY / normalLength
    return normalX, normalY
    
def calculatePerpendicularVector(vectorX, vectorY):
    normalX = -vectorY
    normalY = vectorX
    return normalX, normalY
    
def lengthOfVector(vectorX, vectorY):
    length = math.sqrt((vectorX*vectorX) + (vectorY*vectorY))
    return length
    
def crossProductPositive(vector1X, vector1Y, vector2X, vector2Y):
    return ((vector1X*vector2Y) - (vector1Y*vector2X)) > 0