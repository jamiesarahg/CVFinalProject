import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import math
import statistics as stat

def load_landmark_data(directory, num_images):
    #inputs directory of where landmark data is saved and number of images to load. 
    #outputs a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    landmarks = np.zeros((num_images,8,80))
    for i in range(num_images):
        for j in range(8):
            landmarks[i][j] = np.loadtxt(directory+'/landmarks'+str(i+1)+'-'+str(j+1)+'.txt')
    return landmarks

def show_landmarks_on_images(imgDirectory, landmarks):
    #degugging method for viewing landmark data on the image
    #inputs image directory and the loaded landmark data
    counter=0
    for filename in fnmatch.filter(os.listdir(imgDirectory),'*.tif'):
        file_in = imgDirectory+"/"+filename
        img = cv2.imread(file_in)
        
        #draws circles on the image where the landmarks are for each x and y coordinate pair
        for teeth in landmarks[counter]:
            for j in range(8):
                count=0
                x=0
                y=1
                for i in range(40):
                    cv2.circle(img,(int(landmarks[counter][j][x]),int(landmarks[counter][j][y])),1,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
                    count+=1
                    x=x+2
                    y=y+2
        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('img'+str(counter),small)
        counter+=1

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
    images_out = [0]*len(images)
    for i in range(len(images)):
        img = images[i]
        images_out[i] = preprocess_image(img, kernel = kernel)
        detectEdges(images_out[i])

def detectEdges(img):
    threshold1 = 30
    threshold2 = threshold1*3 #ratio is 3:1 for higher to lower threshold
    sobel = 3
    gradient = False
    edges = cv2.Canny(img, threshold1, threshold2, sobel, L2gradient=gradient)
    canny_result = np.copy(img)
    canny_result[edges.astype(np.bool)]=0
    cv2.imshow('cannyresult',cv2.resize(canny_result,(0,0), fx=.5, fy=.5))

def calculateLandmarkWeights(toothSamples):
    distances = []
    for tooth in toothSamples:
        distancesPerTooth = []
        for i in range(len(tooth)/2):
            distancesPerPoint = []
            x1 = tooth[i * 2]
            y1 = tooth[x1 + 1]
            for j in range(len(tooth)/2):
                x2 = tooth[j * 2]
                y2 = tooth[x2 + 1]
                distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
                distancesPerPoint.append(distance)
            distancesPerTooth.append(distancesPerPoint)
        distances.append(distancesPerTooth)
    

def alignFirstToSecondTooth(tooth1, tooth2, weights):
    xTooth1 = tooth1[0::2]
    yTooth1 = tooth1[1::2]
    xTooth2 = tooth2[0::2]
    yTooth2 = tooth2[1::2]
    numberOfPoints = len(weights)
    x1 = np.vdot(weights, xTooth1)
    x2 = np.vdot(weights, xTooth2)
    y1 = np.vdot(weights, yTooth1)
    y2 = np.vdot(weights, yTooth2)
    z = 0
    c1 = 0
    c2 = 0
    for i in range(numberOfPoints):
        z += weights[i] * (math.pow(xTooth2[i], 2) + math.pow(yTooth2[i], 2))
        c1 += weights[i] * ((xTooth1[i] * xTooth2[i]) + (yTooth1[i] * yTooth2[i]))
        c2 += weights[i] * ((yTooth1[i] * xTooth2[i]) - (xTooth1[i] * yTooth2[i]))
    w = np.sum(weights)
    a = np.array([x2,-y2,w,0],[y2,x2,0,w],[z,0,x2,y2],[0,z,-y2,x2])
    b = np.array([x1,y1,c1,c2])
    transformation = np.linalg.solve(a, b)
    return transformation

if __name__ == '__main__':
    #landmarks=load_landmark_data('_Data/Landmarks/original', 14)
    #show_landmarks_on_images('_Data/Radiographs', landmarks)    
    out = import_images('_Data/Radiographs')
    preprocess_all_images(out)