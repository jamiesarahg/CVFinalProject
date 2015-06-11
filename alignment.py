import cv2
import cv2.cv as cv
import numpy as np
import fnmatch
import math
import statistics as stat
import collections
 
import tests
import tools
import importingAndPreprocessing as prep


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
                
                
                 
                distance = tools.distance(x1,x2,y1,y2)
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
    #weightsOfPoints = weightsOfPoints/np.sum(weightsOfPoints)
    return weightsOfPoints        
    
def alignFirstToSecondTooth(tooth2, tooth1, weights=[1]*40):
    #inputs - tooth1 and tooth2 are landmark data from two different samples of the same tooth
    #weights - output of calculateLandmarkWeights function of the respective tooth
    #weights1 = [x/np.sum(weights) for x in weights]
    #weights = 1

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
        z += weights[i] * (xTooth2[i]**2 + yTooth2[i]**2)
        c1 += weights[i] * ((xTooth1[i] * xTooth2[i]) + (yTooth1[i] * yTooth2[i]))
        c2 += weights[i] * ((yTooth1[i] * xTooth2[i]) - (xTooth1[i] * yTooth2[i]))
    w = np.sum(weights)
    #w = 1

    a = np.array([[x2,(-y2),w,0],[y2,x2,0,w],[z,0,x2,y2],[0,z,(-y2),x2]])
    b = np.array([x1,y1,c1,c2])
    transformation = np.linalg.solve(a, b)

    return transformation
    
def alignSetOf1Tooth(template, toothLandmarks, weights=[1]*40):
    #function to align all samples of one tooth
    '''
    inputs:
    template is the landmark data for which everything should be aligned to
    toothLandmarks is a 2d array of the landmark data for this tooth
    weights is the output of the calculate weights function
    
    Follows Active Shape Model paper
    '''
    output = []
    transformations = []
    for i in range(0, toothLandmarks.shape[0]):

        toothi = toothLandmarks[i]

        transformationMatrix = alignFirstToSecondTooth(toothLandmarks[i], template, weights)
        E = transformLandmarks(transformationMatrix, toothLandmarks[i])
        #img = cv2.imread('_Data/Radiographs/'+str(i+1)+'.tif')
        #tests.plot1toothLandmarkonImage(img, toothLandmarks[0])
        #tests.plot1toothLandmarkonImage(img, toothLandmarks[i])
        #tests.plot1toothLandmarkonImage(img, E)

       
        output.append(E)
        transformations.append(transformationMatrix)
    return output, transformations
        #np.savetxt('_Data/AlignedLandmarksaligned'+str(i+1)+'.txt',E)
        #

def transformLandmarks(transformation, landmarks):
    '''
    transforms the landmarks for one tooth in one image according to the given transformation matrix
    '''
    t = [transformation[2], transformation[3]] * (len(landmarks)/2)
    ax = transformation[0]
    ay = transformation[1]
    M = []
    for j in range(len(landmarks)/2):
       top = ax * landmarks[2*j] - ay * landmarks[2*j+1]
       M.append(top)
       bottom = ay * landmarks[2*j] + ax * landmarks[2*j+1]
       M.append(bottom)
    transformedLandmarks = map(sum, zip(M,t))#M+t
    return transformedLandmarks

def alignmentIteration(landmarks, template):
    '''
    Aligns all of the landmarks from the input landmarks to the template 
    landmarks : all of the landmark data to be aligned
    '''
    newLandmarks = np.zeros([landmarks.shape[0],landmarks.shape[1],landmarks.shape[2]])
    transformations =[]
    for toothNum in range(0,8):
        toothLandmarks = tools.getLandmarksOfTooth(landmarks, toothNum)
        templateData = template[toothNum]
        #weights = calculateLandmarkWeights(toothLandmarks)
        out = alignSetOf1Tooth(templateData,toothLandmarks)
        transformations.append(out[1])
        for i in range(14):
            newLandmarks[i][toothNum] = out[0][i]
    return newLandmarks, transformations
    

def normalize(mean,template):
    '''
    normalizes the calculated mean to the given template
    mean:2d array of shape of (8, 80) of landmark data for each of the 8 teeth
    template:2d array of shape of (8, 80) of landmark data for each of the 8 teeth
    outputs:2d array of shape of (8, 80) of landmark data for each of the 8 teeth of the mean mapped to the template
    '''
    #weights = calculateLandmarkWeights(mean) NEED TO ADD THIS FUNCTIONALITY

    normalized = np.zeros([8,80])
    for toothnum in range(8):
        toothmean = mean[toothnum]
        transformationMatrix = alignFirstToSecondTooth(toothmean,template[toothnum])
        normalized[toothnum] = transformLandmarks(transformationMatrix, toothmean)
    #img = cv2.imread('_Data/Radiographs/1.tif')
    #tests.show_landmarks_on_image(img, normalized, 0)
    return normalized
    
def checkConvergence(transformations):
    for transformationSet in transformations:
        for transformation in transformationSet:
            if (abs(1-transformation[0]) > 10e-4) or (abs(transformation[1]) > 10e-4) or (abs(transformation[2]) > 10e-4) or (abs(transformation[3]) > 10e-4):
                return False
    return True      
    
def checkConvergence2(newLandmarks, oldLandmarks):
    for i in range(newLandmarks.shape[0]):
        for j in range(newLandmarks.shape[1]):
            for k in range(newLandmarks.shape[2]):
                if abs(newLandmarks[i][j][k] - oldLandmarks[i][j][k]) > 10e-10:
                    return False
    return True 
    
def checkConvergence3(newMeans, oldMeans):
    for i in range(newMeans.shape[0]):
        for j in range(newMeans.shape[1]):
            if abs(newMeans[i][j] - oldMeans[i][j]) > 10e-10:
                return False
    return True
    
def alignment(landmarks):
    '''top level alignment function. takees in landmark data and returns aligned landmark data'''
    newLandmarks, transformations = alignmentIteration(landmarks,landmarks[0])
    oldLandmarks = landmarks
    done = False
    count = 0
    oldMeans = np.zeros([8,80])
    while done==False:
        count +=1
        newMeans = tools.calcMean(newLandmarks)
        newMeans = normalize(newMeans, oldLandmarks[0])
        newLandmarks, transformations = alignmentIteration(newLandmarks,newMeans)
        done = checkConvergence3(newMeans, oldMeans)
        if (count == 20):
            done = True
        oldMeans = newMeans
    print 'Number of iterations: '+str(count)
    return newMeans
    
if __name__ == '__main__':
    landmarks = prep.load_landmark_data('_Data/Landmarks/original', 14)    
    means = alignment(landmarks)
    newLandmarks = np.zeros([landmarks.shape[0],landmarks.shape[1],landmarks.shape[2]])
    for i in range(landmarks.shape[0]):
        for j in range(landmarks.shape[1]):
            transformation = alignFirstToSecondTooth(means[j], landmarks[i][j])
            newLandmarks[i][j] = transformLandmarks(transformation, means[j])
    tests.show_landmarks_on_images('_Data/Radiographs', newLandmarks)
