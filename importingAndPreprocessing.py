import cv2
import cv2.cv as cv
import os
import fnmatch
import numpy as np


def test():
    print 'TADA'
    return
    
def load_landmark_data(directory, num_images):
    #inputs directory of where landmark data is saved and number of images to load. 
    #outputs a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    landmarks = np.zeros((num_images,8,80))
    for i in range(num_images):
        for j in range(8):
            landmarks[i][j] = np.loadtxt(directory+'/landmarks'+str(i+1)+'-'+str(j+1)+'.txt')
    return landmarks

def import_images(directory, show=False):
  #  images=np.zeros((14,1600,3023,3)) #array 14(number of images) long with arrays of the image dimentions
    #load images into images array
    images = []
    count=0
    for filename in fnmatch.filter(os.listdir(directory),'*.tif'):
        file_in = directory+"/"+filename
        #images[count] = cv2.imread(file_in)
        images.append(cv2.imread(file_in,0))
        count+=1
    if show:
        for img in images:
            cv2.imshow('img',cv2.resize(img, (0,0), fx=0.25, fy=0.25))
            cv2.waitKey(0)
    return images
    
def claheObject(img, clipLimit=4.0, tileGridSize=(20,15)):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
    return clahe


def preprocess_image(img, kernel = 13, show=False):
    filtered_img = cv2.medianBlur(img, kernel) #filter image to reduce noise
    if show:
        cv2.imshow('filtered', cv2.resize(filtered_img, (0,0), fx=0.25, fy=0.25))
    #histogram_img = cv2.equalizeHist(filtered_img)
    clahe = claheObject(filtered_img, clipLimit=4)
    histogram_img = clahe.apply(filtered_img)
    if show:
        cv2.imshow('hist',cv2.resize(histogram_img, (0,0), fx=0.25, fy=0.25))
        cv2.waitKey(0)
    return histogram_img
    

def preprocess_all_images(images, kernel=13):
    # run code to process all the images
    xgrads = []
    ygrads = []
    for i in range(len(images)):
        img = images[i]
        [xgrad, ygrad] =process_image(img, kernel)
        xgrads.append(xgrad)
        ygrads.append(ygrad)
    return (xgrads, ygrads)
    
def process_image(img, kernel=13):
    #gradients is a tuble of two images: xgrad and ygaad
    imgEdit = preprocess_image(img, kernel = kernel)
    gradients=calculateXYGradient(imgEdit, show=True)
    return gradients

def detectEdges(img, i):
    threshold1 = 30
    threshold2 = threshold1*3 #ratio is 3:1 for higher to lower threshold
    sobel = 3
    gradient = True
    edges = cv2.Canny(img, threshold1, threshold2, sobel, L2gradient=gradient)
    canny_result = np.copy(img)
    canny_result[edges.astype(np.bool)]=0
    cv2.imshow('cannyresult'+str(i),cv2.resize(edges,(0,0), fx=.5, fy=.5))

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
def calculateXYGradient(img, show=False):
        xGradientImages = []
        yGradientImages = []
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


if __name__ == '__main__':
    out = import_images('_Data/Radiographs')
    preprocess_all_images(out)