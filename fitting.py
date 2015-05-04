import cv2
import numpy as np

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
