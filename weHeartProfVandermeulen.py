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
import modeling

if __name__ == '__main__':
    landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)    
    #aligned = alignment.alignment(landmarks)
    #models = modeling.allPCA(aligned[3], 99)
    #calcMean(landmarks)
    #toothSamples = tools.getLandmarksOfTooth(landmarks, 0)
    #weights = alignment.calculateLandmarkWeights(toothSamples)
    #alignment.alignSetOf1Tooth(landmarks,0, weights)
    #img = cv2.imread('_Data/Radiographs/1.tif')
    #E = landmarks[0][0]
    #tests.plot1toothLandmarkonImage(img, E)    
    #transformAll(landmarks)
    #calculateLandmarkWeights(out)
    tests.show_landmarks_on_images('_Data/Radiographs', landmarks)    
    #out = import_images('_Data/Radiographs')
    ##preprocess_all_images(out)
    #images = prep.import_images('_Data/Radiographs', False)
    #prep.calculateXYGradients(images, True)
