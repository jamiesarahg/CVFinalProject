import numpy as np
import cv2
import cv

import importingAndPreprocessing as prep
import tools

def plotPoints(points, image):
    for point in points:
        x, y = point
        cv2.circle(image,(int(x),int(y)),1,cv2.cv.CV_RGB(255, 0, 0),2, 8, 0 )
    cv2.imshow("points", cv2.resize(image, (0,0), fx=0.5, fy=0.5))
    print 'plotted the points'

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
    

def autoInit(landmarks, images):
    shapeOfImages = {}
    avePositionsX = []
    avePositionsY = []
    for i in range(14):
        shape = images[i].shape
        shapeOfImages[i] = shape
    
    for toothNum in range(0,8):
        x=0
        y=0
        toothLandmarks = tools.getLandmarksOfTooth(landmarks, toothNum)
        for i in range(14):
            imshape = shapeOfImages[i]
            print imshape
            for j in range(40):
                x += toothLandmarks[i][j * 2]/imshape[0]
                y += toothLandmarks[i][j * 2 + 1]/imshape[1]
        x = x/560
        y = y/560
        avePositionsX.append(x)
        avePositionsY.append(y)
        
    print avePositionsX, avePositionsY

    
    

def manualInitialization(img):
    """
    This function allows for manual Initialization. 
    Input: a cv loaded image
    outputs: a list of eight points corresponding to the points that the user clicks
    
    This function will display the image that is inputted into the function and the use should then click each of the eight teeth, starting on the top left to the top right, then bottom left to bottom right
    After eight clicks, the image will exit and the list of points will be returned
    """
    #mouse callback function
    global points #declaring these variables as global so they can be accessed in both draw_circle and this function
    points = []

    def draw_circle(event,x,y,flags,param):
        
        if event==cv2.EVENT_LBUTTONDOWN: #
            
            #draws a circle on the image at the point x,y which was recorded from the event
            cv2.circle(img,(x*2,y*2),5,cv2.cv.CV_RGB(255, 255, 255),-1 )
           
           #for debugging 
            print 'x',x
            print 'y',y
            print points
            points.append((x*2, y*2)) #Must multiply by two because imaged is resized by .5Som
            
    #Create a window and bind the function to the window
    cv2.namedWindow("image")
    #Setting draw_circle to be the function run when the mouse is clicked
    cv2.setMouseCallback("image",draw_circle)
    
    while(len(points)<8):
        cv2.imshow("image", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
        
        if cv.WaitKey(15)%0x100==27:break	
    print points
    plotPoints(points, img)
    cv2.destroyAllWindows()
    return points
def manualInitializationAll(images):
    initPointsAll = []
    for i in range(len(images)):
        initPoints = manualInitialization(images[i])
        initPointsAll.append(initPoints)
    print initPointsAll
    return initPointsAll
    
    
    
if __name__ == '__main__':
    #
    #imgs = prep.import_images('_Data/Radiographs', show=False)
    #
    #manualInitialization(imgs[0])
    
    landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)
    #initialization(landmarks)
    
    #images = prep.import_images('_Data/Radiographs/extra', show=False, nbOfPrecedingImages=14)
    #manualInitializationAll(images)
    
    imgs = prep.import_images('_Data/Radiographs/extra', show=False, nbOfPrecedingImages=14)
    autoInit(landmarks,imgs)
    
