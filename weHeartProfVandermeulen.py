import cv2
import cv2.cv as cv
import numpy as np
import fnmatch
import math
import statistics as stat

import tests
import alignment
import importingAndPreprocessing as prep
import tools
import fitting


            
def calculateMeanShape(shapes):
    #shapes is an array of shapes, each consisting of an array containing interleaved x and y coordinates for the respective shape's landmarks
    meanShape = np.zeros(len(shapes[0]))
    for i in range(len(meanShape)):
        meanShape.append(np.mean(shapes[:][i]))
    return meanShape
    
def calculateCovarianceMatrix(alignedShapes, meanShape=None):
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
    #landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)    
    #tests.show_landmarks_on_images('_Data/Radiographs', landmarks)
    #alignment.alignment(landmarks)
    #calcMean(landmarks)
    #toothSamples = tools.getLandmarksOfTooth(landmarks, 0)
    #weights = alignment.calculateLandmarkWeights(toothSamples)
    #alignment.alignSetOf1Tooth(landmarks,0, weights)
    #img = cv2.imread('_Data/Radiographs/1.tif')
    #E = landmarks[0][0]
    #tests.plot1toothLandmarkonImage(img, E)    
    #transformAll(landmarks)
    #calculateLandmarkWeights(out)
    #prep.show_landmarks_on_images('_Data/Radiographs', landmarks)    
    #out = import_images('_Data/Radiographs')
    #preprocess_all_images(out)
    images = prep.import_images('_Data/Radiographs', False)
    fitting.calculateXYGradients(images, True)