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
    
def calculateCovarianceMatrix(alignedShape, meanShape=None):
    #alignedShapes is an array of aligned shapes (one for each image), each consisting of an array containing interleaved x and y coordinates for the respective shape's landmarks
    if meanShape is None:
        meanShape = tools.calcMeanOneTooth(alignedShape)
    covarianceMatrix = np.matrix(np.zeros((len(meanShape),len(meanShape))))

    for i in range(alignedShape.shape[0]):
        deviation = alignedShape[i] - meanShape
        covarianceMatrix = np.add(covarianceMatrix, np.matrix(np.outer(deviation,deviation)))
    
    covarianceMatrix = covarianceMatrix/alignedShape.shape[0]
    return covarianceMatrix, meanShape
    
def PCA(alignedShape, cutOffValue=None):
        
    covarianceMatrix, meanShape = calculateCovarianceMatrix(alignedShape)    
    eigenValues, originalEigenVectors = np.linalg.eigh(covarianceMatrix)
    
    #convert eigenvectors to a more logical format
    eigenVectors = np.zeros([originalEigenVectors.shape[1],originalEigenVectors.shape[0]])
    for i in range(originalEigenVectors.shape[1]):
        eigenVectors[i][:] = originalEigenVectors[:][i]

    #sort eigenvalues and corresponding eigenvectors
    sortedIndicesAscending = eigenValues.argsort()
    sortedIndicesDescending = sortedIndicesAscending[::-1]
    eigenValues = eigenValues[sortedIndicesDescending]
    eigenVectors = eigenVectors[sortedIndicesDescending,:]
    totalVariance = np.sum(eigenValues)
    lastPrincipalComponent = len(sortedIndicesDescending)

    #if no cut-off value is specified, the user needs to choose the last principal component to be returned
    if cutOffValue is None:
        totalPercentageExplainedVariance = 0
        for i in range(len(eigenValues)):
            additionalPercentageExplainedVariance = eigenValues[i]/totalVariance
            totalPercentageExplainedVariance += additionalPercentageExplainedVariance
            print 'Principal component nb ' + str(i+1) + ' explains ' + str(additionalPercentageExplainedVariance*100) + '% out of ' + str(totalPercentageExplainedVariance*100) + '% of the variance explained up until now'
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
            if (totalPercentageExplainedVariance*100) > cutOffValue:
                lastPrincipalComponent = i + 1
                break
    
    #the mean shape along with the most important principal components and their variances will be returned
    return meanShape, eigenValues[:lastPrincipalComponent], eigenVectors[:lastPrincipalComponent,:] 

def allPCA(alignedShapes, cutOffValue=None):
    #alignedShapes is a three dimentional array of the images, each with arrays for the eight teeth, each with the aligned landmark data
    #cutOffValue is the minimum percentage of variance that needs to be explained by the smallest number of principal components that are returned by this function
    models = []
    for i in range(8):
        singleTooth = tools.getLandmarksOfTooth(alignedShapes, i)
        model  = PCA(singleTooth, cutOffValue)
        models.append(model)
    return models
    
def modelInstance(meanShapeLandmarks, principalEigenvectors, b):
    principalEigenvectors = np.transpose(np.array(principalEigenvectors))
    temp = np.zeros([meanShapeLandmarks.shape[0]])
    for i in range(meanShapeLandmarks.shape[0]):
        temp[i] = meanShapeLandmarks[i]
    meanShapeLandmarks = np.array(temp)
    
    print principalEigenvectors.shape
    print b.shape
    dot = np.dot(principalEigenvectors,b)
    temp = np.zeros([dot.shape[0]])
    for i in range(dot.shape[0]):
        temp[i] = dot[i]    
    dot = np.array(temp)
    
    modelLandmarks = meanShapeLandmarks + dot
    return modelLandmarks
    
if __name__ == '__main__':
    landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)  
    images = prep.import_images('_Data/Radiographs', False)  
    aligned = alignment.alignment(landmarks)
    models = allPCA(aligned[3], 99)
    model = models[0]
    b = model[1]
    #b[0] = -883.77828036
    #b[1] = -202.90089261
    for i in range(len(b)):
        b[i] = -1*math.sqrt(b[i])
    instanceLandmarks = modelInstance(model[0], model[2], b)
    tests.show_landmarks_one_tooth_on_image_dynamic(images[0], instanceLandmarks,'model instance landmarks')
