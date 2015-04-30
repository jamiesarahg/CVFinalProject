import cv2
import cv2.cv as cv
import numpy as np
import fnmatch
import math
import statistics as stat

import tests
import importingAndPreprocessing as prep


def getLandmarksOfTooth(landmarks, toothNum):
    Outlandmarks = np.zeros((14,80))
    for i in range(14):
        Outlandmarks[i] = landmarks[i][toothNum][:]
    return Outlandmarks
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
            return
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
    

def alignFirstToSecondTooth(tooth1, tooth2, weights=[1]*40):
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
    a = np.array([[x2,(-y2),1,0],[y2,x2,0,1],[z,0,x2,y2],[0,z,(-y2),x2]])
    b = np.array([x1,y1,c1,c2])
    transformation = np.linalg.solve(a, b)

    return transformation
    
def alignSetOf1Tooth(landmarks, num):
    toothLandmarks = getLandmarksOfTooth(landmarks, num)
    for i in range(1, toothLandmarks.shape[0]):
        xi = toothLandmarks[i]
        transformationMatrix = alignFirstToSecondTooth(toothLandmarks[0], toothLandmarks[i])
        t = [transformationMatrix[3], transformationMatrix[2]] * 40
        ax = transformationMatrix[0]
        ay = transformationMatrix[1]
        M = []
        for j in range(40):
           top = ax * xi[2*j] - ay * xi[2*j+1]
           M.append(top)
           bottom = ay * xi[2*j] + ax * xi[2*j+1]
           M.append(bottom)
        E = M+t
        np.savetxt('_Data/AlignedLandmarks/aligned'+str(i+1)+'.txt',E)
        img = cv2.imread('_Data/Radiographs/'+str(i+1)+'.tif')
        tests.plot1toothLandmarkonImage(img, toothLandmarks[0])
        tests.plot1toothLandmarkonImage(img, toothLandmarks[i])
        tests.plot1toothLandmarkonImage(img, E)
        
        
def transformAll(landmarks):
    print landmarks.shape[1]
    
def calculateMeanShape(shapes):
    #shapes is an array of shapes, each consisting of an array containing interleaved x and y coordinates for the respective shape's landmarks
    meanShape = np.zeros(len(shapes[0]))
    for i in range(len(meanShape)):
        meanShape.append(np.mean(shapes[:][i]))
    return meanShape
    
def calculateCovarianceMatrix(meanShape=None,alignedShapes):
    #alignedShapes is an array of aligned shapes, each consisting of an array containing interleaved x and y coordinates for the respective shape's landmarks
    if meanShape is None:
        meanShape = calculateMeanShape(alignedShapes)
    covarianceMatrix = np.matrix(np.zeros((len(meanShape),len(meanShape))))
    for i in range(len(alignedShapes)):
        deviation = alignedShapes[i] - meanShape
        covarianceMatrix = covarianceMatrix + np.matrix(np.outer(deviation,deviation))
    covarianceMatrix = covarianceMatrix/len(alignedShapes)
    return covarianceMatrix
    
def PCA(alignedShapes,cutOffValue=None):
    #alignedShapes is an array of aligned shapes, each consisting of an array containing interleaved x and y coordinates for the respective shape's landmarks
    #cutOffValue is the minimum percentage of variance that needs to be explained by the smallest number of principal components that are returned by this function
    meanShape = calculateMeanShape(alignedShapes)
    covarianceMatrix = calculateCovarianceMatrix(meanShape,alignedShapes)
    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
    #sort eigenvalues and corresponding eigenvectors
    sortedIndicesAscending = eigenValues.argsort()
    sortedIndicesDescending = sortedIndicesAscending.reverse()
    eigenValues = eigenValues[sortedIndicesDescending]
    eigenVectors = eigenVectors[:,sortedIndicesDescending]
    totalVariance = np.sum(eigenValues)
    lastPrincipalComponent = len(sortedIndicesDescending)
    #if no cut-off value is specified, the user needs to choose the last principal component to be returned
    if cutOffValue is None:
        totalPercentageExplainedVariance = 0
        for i in range(len(eigenValues)):
            additionalPercentageExplainedVariance = eigenValues[i]/totalVariance
            totalPercentageExplainedVariance += additionalPercentageExplainedVariance
            print 'Principal component nb ' + str(i+1) + ' explains ' + str(additionalPercentageExplainedVariance) + '% out of ' + str(totalPercentageExplainedVariance) + '% of the variance explained up until now'
        keyBoardInput = int(input("What would you like the last principal component to be? Please insert its number!"))
        if keyBoardInput < 1 or keyBoardInput > lastPrincipalComponent:
            print 'Invalid number, all principal components will be returned!'
        else:
            lastPrincipalComponent = keyBoardInput
    #if a cut-off value is specified, the minimal number of principal components will be returned that explains at least this percentage of variance
    else:
        totalPercentageExplainedVariance = 0
        for i in range(len(eigenValues)):
            additionalPercentageExplainedVariance = eigenValues[i]/totalVariance
            totalPercentageExplainedVariance += additionalPercentageExplainedVariance
            if totalPercentageExplainedVariance > cutOffValue:
                lastPrincipalComponent = i + 1
                break
    #the mean shape along with the most important principal components and their variances will be returned
    return meanShape, eigenValues[:lastPrincipalComponent], eigenVectors[:,:lastPrincipalComponent]
                

if __name__ == '__main__':
    landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)
    out = getLandmarksOfTooth(landmarks, 0)
    alignSetOf1Tooth(landmarks,0)
    img = cv2.imread('_Data/Radiographs/1.tif')
    E = landmarks[0][0]
    tests.plot1toothLandmarkonImage(img, E)    
    transformAll(landmarks)
    #calculateLandmarkWeights(out)
    #prep.show_landmarks_on_images('_Data/Radiographs', landmarks)    
    #out = import_images('_Data/Radiographs')
    #preprocess_all_images(out)