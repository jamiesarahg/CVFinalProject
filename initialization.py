import numpy as np
import cv2
import cv

import importingAndPreprocessing as prep
import tools


def getAvePosition(toothLandmarks):
    """Calculates average position of x and y coordinates for all samples of a single tooth
        inputs- toothLandmarks: all landmark data across samples for one tooth"""
    x =0
    y =0
    for i in range(14):
        for j in range(40):
            x += toothLandmarks[i][j * 2]
            y += toothLandmarks[i][j * 2 + 1]
            
    x = x/560
    y = y/560
    
    return int(x), int(y)
def showAvePosition(avePositionsX, avePositionsY, extra = True):
    """function to show average position on a picture
        inputs - avePositionsX: list with eight components of average position of x coordinates for each tooth
                 avePositionsY: list with eight components of average position of y coordinates for each tooth
                 extra: if True, uses test set, if false uses training set
                 imgNum: Number of image to display. If extra is True, must be <= 14 and if extra is False must be >=15 and <=30
        """
    imgDirectory = '_Data/Radiographs'
    if extra:
        imgDirectory = '_Data/Radiographs/extra'
    file_in = imgDirectory+'/18'+'.tif'
    img = cv2.imread(file_in) 
    for i in range(8):
        cv2.circle(img,(avePositionsX[i],avePositionsY[i]),3,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
    cv2.imshow('testAvePosition',cv2.resize(img, (0,0), fx=0.25, fy=0.25))
    cv2.waitKey(0)
    

def initialization(landmarks):
    """Main function for initialization of model
        inputs - landmark data for entire training set
        outputs - """
    avePositionsX =[]
    avePositionsY = []
    for toothNum in range(0,8):
        toothLandmarks = tools.getLandmarksOfTooth(landmarks, toothNum)
        avePositionToothX, avePositionToothY = getAvePosition(toothLandmarks)
        avePositionsX.append(avePositionToothX)
        avePositionsY.append(avePositionToothY)
    
    showAvePosition(avePositionsX, avePositionsY)

def manualInitialization(img):
    
    #mouse callback function
    global ix, iy, points
    ix, iy = 0,0
    jx, jy = 0,0
    points = []
    def draw_circle(event,x,y,flags,param):
        print 'in function!!'
        if event==cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),1,cv2.cv.CV_RGB(255, 255, 255),-1 )
            #points.append(x,y)
            ix,iy = x, y

            print 'ix',ix
            print 'iy',iy
            print points
            points.append((ix, iy))
    #Create a window and bind the function to the window
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",draw_circle)
    
    while(len(points)<8):
        cv2.imshow("image", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
        if jx != ix: #or jy != iy:
            points.append((ix, iy))
            print points
        jx = ix
        jy = iy
        
        if cv.WaitKey(15)%0x100==27:break	
    
    cv2.destroyAllWindows()
    print points
    return points

    
if __name__ == '__main__':
    
    imgs = prep.import_images('_Data/Radiographs', show=False)
    
    manualInitialization(imgs[0])
    
    #landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)
    #initialization(landmarks)