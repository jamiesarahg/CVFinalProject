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
    
def searchForToothInImage(cleanImage, compModel, gsModels, nbOfGsTestSamplesPerSide):
    #matchModelToShape(meanShapeLandmarks, principalEigenvalues, principalEigenvectors, shapeLandmarks, landmarkWeights)   
    pass

def searchForTeethInImage(image, componentModels, grayscaleModels, nbOfGsTestSamplesPerSide=20):
    markedImage = copy.deepcopy(image)
    for i in range(componentModels.shape[0]):
        toothLandmarks = searchForToothInImage(image, componentModels[i], grayscaleModels[i], nbOfGsTestSamplesPerSide)
        tests.markLandmarksInImage(markedImage, toothLandmarks)
    return markedImage

def searchForTeethInImages(nbOfGsModelSamplesPerSide=10, nbOfGsTestSamplesPerSide=20, cutOffValue=None, nbOfPrincipalComponents=40):
    #import and preprocess training set images
    trainingImages = prep.import_images('_Data/Radiographs', False)
    trainingImages =  prep.preprocess_all_images(trainingImages, show=False)
    #import and preprocess test set images
    testImages = prep.import_images('_Data/Radiographs/extra', False, 14)
    testImages =  prep.preprocess_all_images(testImages, show=False)
    #import landmarks for training set images
    landmarks = prep.load_landmark_data('_Data/Landmarks/original', 14)
    #align landmarks
    aligned = alignment.alignment(landmarks)
    alignedLandmarks = aligned[3]
    #build component model per shape (one model per tooth)
    componentModels = modeling.allPCA(alignedLandmarks, cutOffValue, nbOfPrincipalComponents)
    #build grayscale model per landmark per shape (one model per tooth landmark)
    grayscaleModels = gs.buildAllGreyscaleModels(landmarks, nbOfGsModelSamplesPerSide, trainingImages, showPoints=False)
    #search for teeth in every test set image
    markedImages = []
    for testImage in testImages:
        markedImage = searchForTeethInImage(testImage, componentModels, grayscaleModels, nbOfGsTestSamplesPerSide)
        markedImages.append(markedImage)
    #return all marked test set images
    return markedImages
        
if __name__ == '__main__':
    searchForTeethInImages(10, 20, None, 40)