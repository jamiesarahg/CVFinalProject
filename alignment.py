import cv2
import cv2.cv as cv
import numpy as np
import fnmatch
import math
import statistics as stat
import collections
 
import tests
import tools



def calculateLandmarkWeights(toothSamples):

    # input- toothSamples - landmark data of one of the teeth but all 14 samples
    l = 0
    y = True
    weightsOfPoints = []
    for landmarkPointNum in range(len(toothSamples[0])/2): #pick a landmark point to calculate the weight for
        variancesPerPoint = []
        for landmarkPointNum2 in range(len(toothSamples[0])/2): #pick a landmark point to compare to the original landmark
            distances = []
            for tooth in toothSamples: #loop through all of samples
                x1 = tooth[landmarkPointNum * 2]
                y1 = tooth[landmarkPointNum * 2 + 1]
                x2 = tooth[landmarkPointNum2 * 2]
                y2 = tooth[landmarkPointNum2 * 2 + 1] 
                
                
                 
                distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)) #calculate the distances of two landmarks in the sample
                distances.append(distance)
            
            if l==1:
                #print distances
                l=l+1 
                l = False
            var = stat.variance(distances) #find the variance of all of the distances from the particular landmark point to all of the other landmarks
            variancesPerPoint.append(var) 
            
        if y:
            #print variancesPerPoint
            y = False 
        pointSum = np.sum(variancesPerPoint) #calculate the sum of all of the variances for the individual landmark point
        pointSumIn = 1/float(pointSum)
        weightsOfPoints.append(pointSumIn)
        #print pointSum
        
    return weightsOfPoints        
    
def alignFirstToSecondTooth(tooth1, tooth2, weights):#=[1]*40):
    #inputs - tooth1 and tooth2 are landmark data from two different samples of the same tooth
    #weights - output of calculateLandmarkWeights function of the respective tooth
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
    w = 1
    print w
    a = np.array([[x2,(-y2),w,0],[y2,x2,0,w],[z,0,x2,y2],[0,z,(-y2),x2]])
    b = np.array([x1,y1,c1,c2])
    transformation = np.linalg.solve(a, b)

    return transformation
    
def alignSetOf1Tooth(template, toothLandmarks, weights):
    output = []
    for i in range(0, toothLandmarks.shape[0]):

        xi = toothLandmarks[i]
        transformationMatrix = alignFirstToSecondTooth(template, toothLandmarks[i], weights)
        t = [transformationMatrix[3], transformationMatrix[2]] * 40
        ax = transformationMatrix[0]
        ay = transformationMatrix[1]
        M = []
        for j in range(40):
           top = ax * xi[2*j] - ay * xi[2*j+1]
           M.append(top)
           bottom = ay * xi[2*j] + ax * xi[2*j+1]
           M.append(bottom)
        E = map(sum, zip(M,t))#M+t
        print E
        #img = cv2.imread('_Data/Radiographs/'+str(i+1)+'.tif')
        #tests.plot1toothLandmarkonImage(img, toothLandmarks[0])
        #tests.plot1toothLandmarkonImage(img, toothLandmarks[i])
        #tests.plot1toothLandmarkonImage(img, E)

       
        output.append(E)
    return output
        #np.savetxt('_Data/AlignedLandmarks/aligned'+str(i+1)+'.txt',E)
        #

def alignmentIteration(landmarks, template, init = False):
    newLandmarks = landmarks
    for toothNum in range(0,8):
        print toothNum
        toothLandmarks = tools.getLandmarksOfTooth(landmarks, toothNum)
        if init==True:
            template = toothLandmarks[0]
        weights = calculateLandmarkWeights(toothLandmarks)
        print 'hey'
        out = alignSetOf1Tooth(template,toothLandmarks, weights)
        print toothNum
        for i in range(14):
            print i
            newLandmarks[i][toothNum] = out[i]
    return newLandmarks
    
def alignment(landmarks):
    new = alignmentIteration(landmarks,None, init=True)
    #tests.show_landmarks_on_images('_Data/Radiographs/', landmarks)
    tests.show_landmarks_on_images('_Data/Radiographs/', new)
    