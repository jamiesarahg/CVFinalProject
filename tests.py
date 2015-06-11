import cv2
import cv2.cv as cv
import os
import fnmatch
def plot1toothLandmarkonImage(img, E):       
        x = 0
        y = 1
        for i in range(40):
                    cv2.circle(img,(int(E[x]),int(E[y])),1,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
                    
                    x=x+2
                    y=y+2
        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('img',small)
        cv2.waitKey(0)
        
def show_landmarks_on_image(img, landmark, count):
        #draws circles on the image where the landmarks are for each x and y coordinate pair
        for teeth in landmark:
            print teeth
            for j in range(8):
                count=0
                x=0
                y=1
                for i in range(40):
                    cv2.circle(img,(int(landmark[j][x]),int(landmark[j][y])),1,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
                    count+=1
                    x=x+2
                    y=y+2
        small = cv2.resize(img, (0,0), fx=0.25, fy=0.25) 
        cv2.imshow('img'+str(count),small)
        cv2.waitKey(0)


def show_landmarks_on_images(imgDirectory, landmarks):
    #degugging method for viewing landmark data on the image
    #inputs image directory and the loaded landmark data
    counter=0
    for i in range(14):
        file_in = imgDirectory+"/"+str(i+1)+'.tif'
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
        cv2.waitKey(0)
        counter+=1
    cv2.destroyAllWindows()
    
def show_landmarks_on_image_dynamic(image, landmarks, name='showing landmarks'):
    #images is an array of images
    #landmarks is a two dimentional array of the eight teeth, each with its landmark data
    for i in range(landmarks.shape[0]):
        for j in range(landmarks.shape[1]/2):
            cv2.circle(image,(int(landmarks[i][2*j]),int(landmarks[i][2*j+1])),1,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
    small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    cv2.imshow(name,small)
    cv2.waitKey(0)