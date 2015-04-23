import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch

def load_landmark_data(directory, num_images):
    landmarks = np.zeros((num_images,8,80))
    for i in range(num_images):
        for j in range(8):
            landmarks[i][j] = np.loadtxt(directory+'/landmarks'+str(i+1)+'-'+str(j+1)+'.txt')
    return landmarks

def show_landmarks_on_images(imgDirectory, landmarks):
    counter=0
    for filename in fnmatch.filter(os.listdir(imgDirectory),'*.tif'):
        file_in = imgDirectory+"/"+filename
        img = cv2.imread(file_in)
        
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

def preprocess_images(directory):
    images=np.zeros((14,1600,3023,3))
    count=0
    for filename in fnmatch.filter(os.listdir(directory),'*.tif'):
        file_in = directory+"/"+filename
        images[count] = cv2.imread(file_in)
        count+=1
    for img in images:
        cv2.imshow('img',img)
    return images

if __name__ == '__main__':
    landmarks=load_landmark_data('_Data/Landmarks/original', 14)
    show_landmarks_on_images('_Data/Radiographs', landmarks)    
