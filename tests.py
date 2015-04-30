import cv2
import cv2.cv as cv
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