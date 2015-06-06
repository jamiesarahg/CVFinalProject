import numpy as np
import cv2

import importingAndPreprocessing as prep
import tools


def getAvePosition(toothLandmarks):
    x =0
    y =0
    for i in range(14):
        for j in range(40):
            x += toothLandmarks[i][j * 2]
            y += toothLandmarks[i][j * 2 + 1]
            
    x = x/560
    y = y/560
    
    return int(x), int(y)
def showAvePosition(avePositionsX, avePositionsY):
    imgDirectory = '_Data/Radiographs/extra'
    file_in = imgDirectory+'/18'+'.tif'
    img = cv2.imread(file_in) 
    for i in range(8):
        cv2.circle(img,(avePositionsX[i],avePositionsY[i]),3,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
    cv2.imshow('testAvePosition',cv2.resize(img, (0,0), fx=0.25, fy=0.25))
    cv2.waitKey(0)
    

def initialization(landmarks):
    avePositionsX =[]
    avePositionsY = []
    for toothNum in range(0,8):
        toothLandmarks = tools.getLandmarksOfTooth(landmarks, toothNum)
        avePositionToothX, avePositionToothY = getAvePosition(toothLandmarks)
        avePositionsX.append(avePositionToothX)
        avePositionsY.append(avePositionToothY)
    
    showAvePosition(avePositionsX, avePositionsY)
        
        
    pass
    
if __name__ == '__main__':
    landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)
    initialization(landmarks)