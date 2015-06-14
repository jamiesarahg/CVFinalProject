import cv2
import numpy as np
import math
import importingAndPreprocessing as prep
import tests
import alignment
import modeling
import tools
import fitting
import grayscaleModeling as gs
import copy

def getInitialToothCenterInImage(toothNb, cleanImage):
    if toothNb is 0:
        
    
def getInitialToothLandmarksInImage(toothNb, cleanImage, meanToothLandmarks):
    xCenter, yCenter = tools.calcCenter(meanToothLandmarks)
    initXCenter, initYCenter = getInitialToothCenterInImage(cleanImage, toothNb)
    xTranslation = initXCenter - xCenter
    yTranslation = initYCenter - yCenter
    initialToothLandmarks = copy.deepcopy(meanToothLandmarks)
    for i in range(len(initialToothLandmarks)/2):
        initialToothLandmarks[2*i] = xTranslation + meanToothLandmarks[2*i]
        initialToothLandmarks[2*i+1] = yTranslation + meanToothLandmarks[2*i+1]
    return initialToothLandmarks

def searchConvergenceCheck(oldB, newB, oldTransformation, newTransformation):
    converged = tools.valuesConvergenceCheck(oldB, newB) and tools.valuesConvergenceCheck(oldTransformation, newTransformation)
    return converged

def searchForToothInImage(toothNb, cleanImage, landmarkWeights, compModel, gsModelMeans, gsModelCovarMatrices, nbOfGsTestSamplesPerSide):
    meanToothLandmarks = compModel[0]
    principalComponentVariances = compModel[1]
    principalComponents = compModel[2]
    
    prevModelLandmarks = getInitialToothLandmarksInImage(toothNb, cleanImage, meanToothLandmarks)
    prevB = np.transpose(np.matrix(np.zeros([len(principalComponentVariances)])))
    prevTransformation = (1,0,0,0)
    
    converged = False
    while not converged:
        intermediateToothLandmarks = gs.calculateNewLandmarksForToothInImage(prevModelLandmarks, nbOfGsTestSamplesPerSide, gsModelMeans, gsModelCovarMatrices, cleanImage)
        nextTransformation, nextB, nextModelLandmarks = fitting.matchModelToShape(meanToothLandmarks, principalComponentVariances, principalComponents, intermediateToothLandmarks, landmarkWeights)
        converged = searchConvergenceCheck(prevB, nextB, prevTransformation, nextTransformation)
        prevModelLandmarks = nextModelLandmarks
        prevB = nextB
        prevTransformation = nextTransformation
    return nextTransformation, nextB, nextModelLandmarks

def searchForTeethInImage(image, landmarkWeights, componentModels, grayscaleModelMeans, grayscaleModelCovarianceMatrices, nbOfGsTestSamplesPerSide=20):
    markedImage = copy.deepcopy(image)
    for i in range(componentModels.shape[0]):
        toothFeatures = searchForToothInImage(i, image, landmarkWeights[i], componentModels[i], grayscaleModelMeans[i], grayscaleModelCovarianceMatrices[i], nbOfGsTestSamplesPerSide)
        toothLandmarks = toothFeatures[2]
        tests.markLandmarksInImage(markedImage, toothLandmarks)
    return markedImage

def searchForTeethInImages(nbOfGsModelSamplesPerSide=10, nbOfGsTestSamplesPerSide=20, cutOffValue=None, nbOfPrincipalComponents=40):
    #import and preprocess training set images
    trainingImages = prep.import_images('_Data/Radiographs', False)
    trainingImages =  prep.preprocess_all_images(trainingImages, False)
    #import and preprocess test set images
    testImages = prep.import_images('_Data/Radiographs/extra', False, 14)
    testImages =  prep.preprocess_all_images(testImages, False)
    #import landmarks for training set images
    landmarks = prep.load_landmark_data('_Data/Landmarks/original', 14)
    #calculate landmark weights
    landmarkWeights = alignment.calculateLandmarkWeightsForAllTeeth(landmarks)
    #align landmarks
    aligned = alignment.alignment(landmarks, landmarkWeights)
    alignedLandmarks = aligned[3]
    #build component model per shape (one model per tooth)
    componentModels = modeling.allPCA(alignedLandmarks, cutOffValue, nbOfPrincipalComponents)
    #build grayscale model per landmark per shape (one model per tooth landmark)
    grayscaleModelCovarianceMatrices, grayscaleModelMeans = gs.buildAllGreyscaleModels(landmarks, nbOfGsModelSamplesPerSide, trainingImages, showPoints=False)
    #search for teeth in every test set image
    markedImages = []
    for testImage in testImages:
        markedImage = searchForTeethInImage(testImage, landmarkWeights, componentModels, grayscaleModelMeans, grayscaleModelCovarianceMatrices, nbOfGsTestSamplesPerSide)
        markedImages.append(markedImage)
    #return all marked test set images
    return markedImages
        
if __name__ == '__main__':
    searchForTeethInImages(10, 20, None, 40)