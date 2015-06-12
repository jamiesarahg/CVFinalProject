import cv2
import cv2.cv as cv
import numpy as np
import fnmatch
import math
import statistics as stat
 
import fitting
import tests
import tools
import importingAndPreprocessing as prep


def calculateLandmarkWeightsForTooth(toothSamples):
    #toothSamples is a two dimentional array of the images, each with the landmark data for the same one tooth
    weightsOfPoints = []
    for landmarkPointNum in range(len(toothSamples[0])/2): #pick a landmark point to calculate the weight for
        variancesPerPoint = []
        for landmarkPointNum2 in range(len(toothSamples[0])/2): #pick a landmark point to compare to the original landmark
            distances = []
            for tooth in toothSamples: #loop through all the samples (all the images)
                x1 = tooth[landmarkPointNum * 2]
                y1 = tooth[landmarkPointNum * 2 + 1]
                x2 = tooth[landmarkPointNum2 * 2]
                y2 = tooth[landmarkPointNum2 * 2 + 1] 
                distance = tools.distance(x1,x2,y1,y2)
                distances.append(distance)
            var = stat.variance(distances) #find the variance of all of the distances from the particular landmark point to all of the other landmarks
            variancesPerPoint.append(var) 
        pointSum = np.sum(variancesPerPoint) #calculate the sum of all of the variances for the individual landmark point
        pointSumInv = 1/float(pointSum)
        weightsOfPoints.append(pointSumInv)
    #normalizedWeights = fitting.normalizeValues(weightsOfPoints)
    #weightsOfPoints is a one dimentional array of the weights for the points of one tooth (size = number of landmarks for this tooth)
    return weightsOfPoints 
    
def calculateLandmarkWeightsForAllTeeth(allToothSamples):
    #allToothSamples is a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    allToothWeights = []
    for i in range(allToothSamples.shape[1]):
        #toothSamples = allToothSamples[:][i][:]
        toothSamples = tools.getLandmarksOfTooth(allToothSamples, i)
        toothWeights = calculateLandmarkWeightsForTooth(toothSamples)
        allToothWeights.append(toothWeights)
    #allToothWeights is a two dimentional array of all the teeth, each with an array of the weights for the points of this tooth (size = number of landmarks for this tooth)
    return allToothWeights
    
def alignFirstToSecondTooth(tooth2, tooth1, weights):
    #inputs - tooth1 and tooth2 are landmark data from two different samples of the same tooth
    #tooth2 is an array of landmark data for one sample
    #tooth1 is an array of landmark data for another sample
    #weights is a one dimentional array of the weights for the points of this tooth (size = number of landmarks for this tooth)
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
    a = np.array([[x2,(-y2),w,0],[y2,x2,0,w],[z,0,x2,y2],[0,z,(-y2),x2]])
    b = np.array([x1,y1,c1,c2])
    transformation = np.linalg.solve(a, b)
    return transformation
    
def alignSetOf1Tooth(template, toothLandmarks, weights):
    '''
    inputs:
    template is the landmark data for which everything should be aligned to
    toothLandmarks is a 2d array of the landmark data for this tooth
    weights is the output of the calculate weights function
    
    Follows Active Shape Model paper
    '''
    #template is an array of landmark data to which the toothLandmarks should be aligned
    #toothLandmarks is a two dimentional array of the images, each with the landmark data for the same one tooth
    #weights is a one dimentional array of the weights for the points of this tooth (size = number of landmarks for this tooth)
    allTransformedLandmarks = []
    transformations = []
    for i in range(toothLandmarks.shape[0]):
        toothi = toothLandmarks[i]
        transformationMatrix = alignFirstToSecondTooth(toothi, template, weights)
        transformedLandmarksForOneTooth = transformLandmarks(transformationMatrix, toothi)
        allTransformedLandmarks.append(transformedLandmarksForOneTooth)
        transformations.append(transformationMatrix)
    return allTransformedLandmarks, transformations

def transformLandmarks(transformation, landmarks):
    '''
    transforms the landmarks for one tooth in one image according to the given transformation matrix
    '''
    #transformation is an array of transformation data (which is to be used to transform the landmarks)
    #landmarks is an array of landmark data for one sample
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

def alignmentIteration(landmarks, templates, weights):
    '''
    Aligns all of the landmarks from the input landmarks to the template 
    landmarks : all of the landmark data to be aligned
    '''
    #landmarks is a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    #templates is a two dimentional array of the teeth, each with template landmark data to which the respective teeth should be aligned
    #weights is a two dimentional array of all the teeth, each with an array of the weights for the points of this tooth (size = number of landmarks for this tooth)
    newLandmarks = np.zeros([landmarks.shape[0],landmarks.shape[1],landmarks.shape[2]])
    transformations =[]
    for toothNum in range(landmarks.shape[1]):
        toothLandmarks = tools.getLandmarksOfTooth(landmarks, toothNum)
        #toothLandmarks = landmarks[:][toothNum][:]
        out = alignSetOf1Tooth(templates[toothNum],toothLandmarks,weights[toothNum])
        transformations.append(out[1])
        for i in range(14):
            newLandmarks[i][toothNum] = out[0][i]
    return newLandmarks, transformations
    

def normalize(means,templates,weights):
    '''
    normalizes the calculated mean to the given template
    mean:2d array of shape of (8, 80) of landmark data for each of the 8 teeth
    template:2d array of shape of (8, 80) of landmark data for each of the 8 teeth
    outputs:2d array of shape of (8, 80) of landmark data for each of the 8 teeth of the mean mapped to the template
    '''
    #means is a two dimentional array of the teeth, each with mean landmark data (across the images)
    #templates is a two dimentional array of the teeth, each with landmark data based on which the means should be normalized
    #weights is a two dimentional array of all the teeth, each with an array of the weights for the points of this tooth (size = number of landmarks for this tooth)
    normalized = np.zeros([means.shape[0],means.shape[1]])
    for toothnum in range(means.shape[0]):
        toothmean = means[toothnum]
        transformationMatrix = alignFirstToSecondTooth(toothmean,templates[toothnum],weights[toothnum])
        normalized[toothnum] = transformLandmarks(transformationMatrix, toothmean)
    return normalized
    
def checkConvergence(newMeans, oldMeans):
    #newMeans is a two dimentional array of the teeth, each with new mean landmark data (across the images)
    #oldMeans is a two dimentional array of the teeth, each with old mean landmark data (across the images)
    for i in range(newMeans.shape[0]):
        for j in range(newMeans.shape[1]):
            if abs(newMeans[i][j] - oldMeans[i][j]) > 10e-10:
                return False
    return True
    
def alignment(landmarks, weights=None):
    '''top level alignment function. takees in landmark data and returns aligned landmark data'''
    #landmarks is a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    if weights is None:
        weights = np.ones([landmarks.shape[1],landmarks.shape[2]/2])
    newLandmarks, transformations = alignmentIteration(landmarks,landmarks[0],weights)
    oldLandmarks = landmarks
    done = False
    count = 0
    oldMeans = np.zeros([landmarks.shape[1],landmarks.shape[2]])
    while done==False:
        count +=1
        newMeans = tools.calcMean(newLandmarks)
        newMeans = normalize(newMeans, oldLandmarks[0], weights)
        newLandmarks, transformations = alignmentIteration(newLandmarks,newMeans,weights)
        done = checkConvergence(newMeans, oldMeans)
        if (count == 20):
            done = True
        oldMeans = newMeans
    print 'Number of iterations: '+str(count)
    transformedMeans = np.zeros([landmarks.shape[0],landmarks.shape[1],landmarks.shape[2]])
    transformations = np.zeros([landmarks.shape[0],landmarks.shape[1],4])
    for i in range(landmarks.shape[0]):
        for j in range(landmarks.shape[1]):
            transformation = alignFirstToSecondTooth(newMeans[j], oldLandmarks[i][j], weights[j])
            transformations[i][j] = transformation
            transformedMeans[i][j] = transformLandmarks(transformation, newMeans[j])
    #transformations is a three dimentional array of the images, each with arrays for the eight teeth, each with the transformation data (which aligns the respective mean landmark data with the old landmark data)
    #newMeans is a two dimentional array of the teeth, each with mean landmark data (across the images)
    #finalLandmarks is a three dimentional array of the images, each with arrays for the eight teeth, each with the new landmark data (which is the mean landmark data that is aligned with the old landmark data)
    return newMeans, transformations, transformedMeans, newLandmarks
    
if __name__ == '__main__':
    landmarks = prep.load_landmark_data('_Data/Landmarks/original', 14)
    images = prep.import_images('_Data/Radiographs', False)
    weights = calculateLandmarkWeightsForAllTeeth(landmarks)
    withWeights = alignment(landmarks, weights)
    '''#plot the mean teeth (calculated based on real weights) on the first image
    tests.show_landmarks_on_image_dynamic(images[0], withWeights[0],'mean shapes with weights')
    withoutWeights = alignment(landmarks)
    #plot the mean teeth (calculated based on unit vector as weights) on the first image
    tests.show_landmarks_on_image_dynamic(images[0], withoutWeights[0],'mean shapes without weights')
    cv2.destroyAllWindows()'''
    tests.show_landmarks_on_images('_Data/Radiographs', withWeights[2])