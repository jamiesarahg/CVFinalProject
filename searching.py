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
import initialization as init

'''this function does not perform as expected for some reason'''
def getAutoInitialTeethCentersInImage(targetImage, landmarks, trainingImages):
    initialRelativeXPositions, initialRelativeYPositions = init.autoInit(landmarks, trainingImages)
    initialAbsoluteXPositions = copy.deepcopy(initialRelativeXPositions)
    initialAbsoluteYPositions = copy.deepcopy(initialRelativeYPositions)
    for toothNb in range(landmarks.shape[1]):
        initialAbsoluteXPositions[toothNb] = initialRelativeXPositions[toothNb] * targetImage.shape[1]
        initialAbsoluteYPositions[toothNb] = initialRelativeYPositions[toothNb] * targetImage.shape[0]
    print 'x relative', initialRelativeXPositions
    print 'y relative', initialRelativeYPositions
    print 'x absolute', initialAbsoluteXPositions
    print 'y absolute', initialAbsoluteXPositions
    return initialAbsoluteXPositions, initialAbsoluteYPositions
    
def getManualInitialTeethCentersInImage(targetImage, landmarks, trainingImages):
    points = init.manualInitialization(targetImage)
    xCoordinates = []
    yCoordinates = []
    for point in points:
        xCoordinates.append(point[0])
        yCoordinates.append(point[1])
    return xCoordinates, yCoordinates
        
def getInitialToothLandmarksInImage(targetImage, meanToothLandmarks, initXCenter, initYCenter):
    xCenter, yCenter = tools.calcCenter(meanToothLandmarks)
    xTranslation = initXCenter - xCenter
    yTranslation = initYCenter - yCenter
    initialToothLandmarks = copy.deepcopy(meanToothLandmarks)
    for i in range(len(initialToothLandmarks)/2):
        initialToothLandmarks[2*i] = xTranslation + meanToothLandmarks[2*i]
        initialToothLandmarks[2*i+1] = yTranslation + meanToothLandmarks[2*i+1]
    return initialToothLandmarks
    
def getInitialTeethLandmarksInImage(targetImage, meanTeethLandmarks, allLandmarks, trainingImages):
    #initialXCenters, initialYCenters = getAutoInitialTeethCentersInImage(targetImage, allLandmarks, trainingImages)
    initialXCenters, initialYCenters = getManualInitialTeethCentersInImage(targetImage, allLandmarks, trainingImages)
    initialTeethLandmarks = copy.deepcopy(meanTeethLandmarks)
    for toothNb in range(len(meanTeethLandmarks)):
        initialTeethLandmarks[toothNb] = getInitialToothLandmarksInImage(targetImage, meanTeethLandmarks[toothNb], initialXCenters[toothNb], initialYCenters[toothNb])
    return initialTeethLandmarks
    
def searchConvergenceCheck(oldB, newB, oldTransformation, newTransformation):
    converged = tools.valuesConvergenceCheck(oldB, newB) and tools.valuesConvergenceCheck(oldTransformation, newTransformation)
    return converged

def searchForToothInImage(cleanImage, initialToothLandmarks, landmarkWeights, compModel, gsModelMeans, gsModelCovarMatrices, nbOfGsTestSamplesPerSide, show=False):
    meanToothLandmarks = compModel[0]
    principalComponentVariances = compModel[1]
    principalComponents = compModel[2]
    
    prevModelLandmarks = initialToothLandmarks
    prevB = np.transpose(np.matrix(np.zeros([len(principalComponentVariances)])))
    prevTransformation = (1,0,0,0)
    
    converged = False
    pleaseStop = False
    while not (converged or pleaseStop):
        try:
            if show:
                tests.show_landmarks_one_tooth_on_image_dynamic(cleanImage, prevModelLandmarks, name='intermediate search results for one tooth', waitkey=True)
            intermediateToothLandmarks = gs.calculateNewLandmarksForToothInImage(prevModelLandmarks, nbOfGsTestSamplesPerSide, gsModelMeans, gsModelCovarMatrices, cleanImage)
            nextTransformation, nextB, nextModelLandmarks = fitting.matchModelToShape(meanToothLandmarks, principalComponentVariances, principalComponents, intermediateToothLandmarks, landmarkWeights)
            converged = searchConvergenceCheck(prevB, nextB, prevTransformation, nextTransformation)
            prevModelLandmarks = nextModelLandmarks
            prevB = nextB
            prevTransformation = nextTransformation
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()
    return nextTransformation, nextB, nextModelLandmarks

def searchForTeethInImage(image, initialTeethLandmarks, landmarkWeights, componentModels, grayscaleModelMeans, grayscaleModelCovarianceMatrices, nbOfGsTestSamplesPerSide=20, show=False):
    markedImage = copy.deepcopy(image)
    for i in range(len(componentModels)):
        toothFeatures = searchForToothInImage(image, initialTeethLandmarks[i], landmarkWeights[i], componentModels[i], grayscaleModelMeans[i], grayscaleModelCovarianceMatrices[i], nbOfGsTestSamplesPerSide, show)
        toothLandmarks = toothFeatures[2]
        tests.markLandmarksInImage(markedImage, toothLandmarks)
    return markedImage

def searchForTeethInImages(nbOfGsModelSamplesPerSide=10, nbOfGsTestSamplesPerSide=20, cutOffValue=None, nbOfPrincipalComponents=40, show=False):
    #import and preprocess training set images
    trainingImages = prep.import_images('_Data/Radiographs', False)
    trainingImages =  prep.preprocess_all_images(trainingImages, False)
    #import and preprocess test set images
    testImages = prep.import_images('_Data/Radiographs/test', False, 14)
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
    meanTeethLandmarks = [model[0] for model in componentModels]
    for testImage in testImages:
        initialTeethLandmarks = getInitialTeethLandmarksInImage(testImage, meanTeethLandmarks, landmarks, trainingImages)
        markedImage = searchForTeethInImage(testImage, initialTeethLandmarks, landmarkWeights, componentModels, grayscaleModelMeans, grayscaleModelCovarianceMatrices, nbOfGsTestSamplesPerSide, show)
        markedImages.append(markedImage)
    #return all marked test set images
    return markedImages
        
if __name__ == '__main__':
    markedImages = searchForTeethInImages(20, 80, cutOffValue=99.99, nbOfPrincipalComponents=None, show=True)
    for img in markedImages:
        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('search results',small)
        cv2.waitKey(0)
    cv2.destroyAllWindows()