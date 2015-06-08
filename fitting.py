import cv2
import numpy as np
import math
import importingAndPreprocessing as prep

def calculateAllLandmarkNormals(allLandmarks):
    #landmarks is a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    allLandmarkNormals = np.zeros((allLandmarks.shape[0],allLandmarks.shape[1],allLandmarks.shape[2]))
    #for each image
    for i in range(allLandmarks.shape[0]):
        #for each tooth
        for j in range(allLandmarks.shape[1]):
            #landmark normals for all landmarks in one shape (tooth j in image i)
            allLandmarkNormals[i][j] = calculateLandmarkNormals(allLandmarks[i][j])
    return allLandmarkNormals

def calculateLandmarkNormals(shapeLandmarks):
    #shapeLandmarks is an  array containing interleaved x and y coordinates for all landmarks of one shape
    normals = []
    loopRange = len(shapeLandmarks)/2
    for i in range(loopRange):
        centerX = shapeLandmarks[2*i]
        centerY = shapeLandmarks[2*i+1]
        if(i==0):
            prevX = shapeLandmarks[2*(loopRange-1)]
            prevY = shapeLandmarks[2*(loopRange-1)+1]
        else:
            prevX = shapeLandmarks[2*(i-1)]
            prevY = shapeLandmarks[2*(i-1)+1]
        if(i==loopRange-1):
            nextX = shapeLandmarks[0]
            nextY = shapeLandmarks[1]
        else:
            nextX = shapeLandmarks[2*(i+1)]
            nextY = shapeLandmarks[2*(i+1)+1]
        normalX, normalY = calculateLandmarkNormal(prevX, prevY, centerX, centerY, nextX, nextY)
        normals.extend([normalX, normalY])
    return normals
    
def calculateLandmarkNormal(prevX, prevY, centerX, centerY, nextX, nextY):
    #calculate the first surface vector
    prevVectorX = prevX-centerX
    prevVectorY = prevY-centerY
    #calculate the first surface normal
    n1x, n1y = calculatePerpendicularVector(prevVectorX, prevVectorY)
    #make sure this surface normal is pointing outwards
    if(crossProductPositive(prevVectorX, prevVectorY, n1x, n1y) is True):
        n1x = -n1x
        n1y = -n1y
    #normalize this surface normal
    n1length = lengthOfVector(n1x, n1y)
    n1x = n1x / n1length
    n1y = n1y / n1length
    #calculate the second surface vector
    nextVectorX = nextX-centerX
    nextVectorY = nextY-centerY
    #calculate the second surface normal
    n2x, n2y = calculatePerpendicularVector(nextVectorX, nextVectorY)
    #make sure this surface normal is pointing outwards
    if(crossProductPositive(nextVectorX, nextVectorY, n2x, n2y) is False):
        n2x = -n2x
        n2y = -n2y
    #normalize this surface normal
    n2length = lengthOfVector(n2x, n2y)
    n2x = n2x / n2length
    n2y = n2y / n2length
    #the required landmark normal is just the sum of the two calculated surface normals
    normalX = n1x + n2x
    normalY = n1y + n2y
    normalLength = lengthOfVector(normalX, normalY)
    normalX = normalX / normalLength
    normalY = normalY / normalLength
    return normalX, normalY
    
def calculatePerpendicularVector(vectorX, vectorY):
    normalX = -vectorY
    normalY = vectorX
    return normalX, normalY
    
def lengthOfVector(vectorX, vectorY):
    length = math.sqrt((vectorX*vectorX) + (vectorY*vectorY))
    return length
    
def crossProductPositive(vector1X, vector1Y, vector2X, vector2Y):
    return ((vector1X*vector2Y) - (vector1Y*vector2X)) > 0
    
def pointsOnLandmarkNormal(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, nbOfPointsPerSide):
    #returns an ordered array containing interleaved x and y coordinates for all desired points on the landmark normal (on both sides of the surface)
    leftPointsReversed = []
    rightPoints = []
    for i in range(nbOfPointsPerSide):
        leftPointX = landmarkX - (nbOfPointsPerSide-i)*landmarkNormalX
        leftPointY = landmarkY - (nbOfPointsPerSide-i)*landmarkNormalY
        leftPointsReversed.extend([leftPointX, leftPointY])
        rightPointX = landmarkX + (i+1)*landmarkNormalX
        rightPointY = landmarkY + (i+1)*landmarkNormalY
        rightPoints.extend([rightPointX, rightPointY])
    points = leftPointsReversed
    points.extend([landmarkX, landmarkY])
    points.extend(rightPoints)
    return points
    
def pointsToPixels(points):
    #points is an array containing interleaved x and y coordinates for all points for which the pixel coordinates are returned (in the same order)
    pixels = []
    for i in range(len(points)/2):
        pixelX = round(points[2*i])
        print pixelX
        pixelX = int(pixelX)
        pixelY = round(points[2*i+1])
        pixelY = int(pixelY)
        pixels.extend([pixelX, pixelY])
    return pixels
    
def calculateDerivatives(pixelValues):
    derivativeValues = []
    for i in range(len(pixelValues)-1):
        derivative = pixelValues[i+1] - pixelValues[i]
        derivativeValues.append(derivative)
    return derivativeValues
    
def buildDerivativeGrayscaleModel(landmarks, landmarkNormals, nbOfSamplesPerSide, grayscaleImages):
    #gradientGreyscaleImages is an array of all the images, after being converted into a gradient greyscale, in the training set
    #landmarks (and corresponding landmarkNormals) is an array of interleaved x and y coordinates for the same landmark across all these images
    #pixels is an array (with size equal to gradientGreyscaleImages) of arrays of interleaved x and y coordinates for the corresponding pixel set
    allPixels = []
    for i in range(len(grayscaleImages)):
        x = 2*i
        y = 2*i+1
        points = pointsOnLandmarkNormal(landmarks[x], landmarks[y], landmarkNormals[x], landmarkNormals[y], nbOfSamplesPerSide)
        pixels = pointsToPixels(points)
        allPixels.append(pixels)
    #retrieve the derivatives for each pixelset per image and store it in an (image X derivativeValues) array
    totalDerivativeValues = []
    for i in range(len(grayscaleImages)):
        pixelValues = []
        for j in range(len(allPixels[0])/2):
            pixelValue = grayscaleImages[i][allPixels[i][2*j]][allPixels[i][2*j+1]]
            pixelValues.append(pixelValue)
        derivativeValues = calculateDerivatives(pixelValues)
        derivativeValues = np.array(derivativeValues) / np.sum(derivativeValues)
        totalDerivativeValues.append(derivativeValues)
    #calculate mean array
    meanDerivativeValues = np.zeros(len(totalDerivativeValues[0]))
    for derivativeValueArray in totalDerivativeValues:
        meanDerivativeValues += derivativeValueArray
    meanDerivativeValues = meanDerivativeValues / len(grayscaleImages)
    #calculate covariance matrix
    covarianceMatrix = np.matrix(np.zeros((len(meanDerivativeValues),len(meanDerivativeValues))))
    for i in range(len(grayscaleImages)):
        deviation = totalDerivativeValues[i] - meanDerivativeValues
        covarianceMatrix = np.add(covarianceMatrix, np.matrix(np.outer(deviation,deviation)))
    covarianceMatrix = covarianceMatrix/len(grayscaleImages)
    return covarianceMatrix, meanDerivativeValues
    
def buildRegularGrayscaleModel(landmarks, landmarkNormals, nbOfSamplesPerSide, gradientGrayscaleImages):
    #gradientGreyscaleImages is an array of all the images, after being converted into a gradient greyscale, in the training set
    #landmarks (and corresponding landmarkNormals) is an array of interleaved x and y coordinates for the same landmark across all these images
    #pixels is an array (with size equal to gradientGreyscaleImages) of arrays of interleaved x and y coordinates for the corresponding pixel set
    allPixels = []
    for i in range(len(gradientGrayscaleImages)):
        x = 2*i
        y = 2*i+1
        points = pointsOnLandmarkNormal(landmarks[x], landmarks[y], landmarkNormals[x], landmarkNormals[y], nbOfSamplesPerSide)
        pixels = pointsToPixels(points)
        allPixels.append(pixels)
    #retrieve the value for each pixelset per image and store it in a (image X pixelvalues) array
    totalPixelValues = []
    for i in len(gradientGrayscaleImages):
        pixelValues = []
        absoluteSum = 0
        for j in range(len(pixels[0])/2):
            pixelValue = gradientGrayscaleImages[i][allPixels[i][2*j]][allPixels[i][2*j+1]]
            pixelValues.append(pixelValue)
            absoluteSum += abs(pixelValue)
        pixelValues = np.array(pixelValues) / absoluteSum
        totalPixelValues.append(pixelValues)
    #calculate mean array
    meanPixelValues = np.zeros(len(pixelValues[0]))
    for pixelValueArray in totalPixelValues:
        meanPixelValues += pixelValueArray
    meanPixelValues = meanPixelValues / len(gradientGrayscaleImages)
    #calculate covariance matrix
    covarianceMatrix = np.matrix(np.zeros((len(meanPixelValues),len(meanPixelValues))))
    for i in range(len(gradientGrayscaleImages)):
        deviation = totalPixelValues[i] - meanPixelValues
        covarianceMatrix = np.add(covarianceMatrix, np.matrix(np.outer(deviation,deviation)))
    covarianceMatrix = covarianceMatrix/len(gradientGrayscaleImages)
    return covarianceMatrix, meanPixelValues
    
def calculateNewLandmarkWithDerivativeGrayscaleModel(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, nbOfSamplesPerSide, modelMean, modelCovarMatrix, grayscaleImage):
    m = nbOfSamplesPerSide
    k = (len(modelMean)-1)/2
    if(m <= k):
        print 'M is not larger than k!!'
    #retrieve the pixelcoordinates for all the necessary pixels on the landmark normal (= 2*m + 1 pixels)
    points = pointsOnLandmarkNormal(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, m)
    allPixels = pointsToPixels(points)
    allPixelValues = []
    #retrieve the pixelvalues for all pixels
    for i in range(len(allPixels)/2):
        pixelValue = grayscaleImage[allPixels[2*i]][allPixels[2*i+1]]
        allPixelValues.append(pixelValue)
    #retrieve the derivatives for all pixelvalues
    allDerivativeValues = calculateDerivatives(allPixelValues)
    #construct 2*(m-k)+1 samples to compare to the greyscale model
    fitValues = []
    for i in range(2*(m-k)+1):
        sampleDerivatives = allDerivativeValues[i:(i+2*k)]
        sampleDerivatives = np.array(sampleDerivatives) / np.sum(sampleDerivatives)
        #compare sample to greyscale model
        fitMeasure = mahalanobisDistance(sampleDerivatives, modelMean, modelCovarMatrix)
        fitValues.append(fitMeasure)
    #get index of best fitting point (smallest fit value)
    sortedFitIndices = fitValues.argsort()
    indexOfBestPixel = sortedFitIndices[0] + k
    #get pixelcoordinates of best fitting point
    pixelXIndex = 2*indexOfBestPixel
    pixelYIndex = pixelXIndex + 1
    return allPixels[pixelXIndex], allPixels[pixelYIndex]
    
def calculateNewLandmarkWithRegularGrayscaleModel(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, nbOfSamplesPerSide, modelMean, modelCovarMatrix, gradientGrayscaleImage):
    m = nbOfSamplesPerSide
    k = (len(modelMean)-1)/2
    if(m <= k):
        print 'M is not larger than k!!'
    #retrieve the pixelcoordinates for all the necessary pixels on the landmark normal (= 2*m + 1 pixels)
    points = pointsOnLandmarkNormal(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, m)
    allPixels = pointsToPixels(points)
    allPixelValues = []
    #retrieve the pixelvalues for all pixels
    for i in range(len(allPixels)/2):
        pixelValue = gradientGrayscaleImage[allPixels[2*i]][allPixels[2*i+1]]
        allPixelValues.append(pixelValue)
    #construct 2*(m-k)+1 samples to compare to the greyscale model
    fitValues = []
    for i in range(2*(m-k)+1):
        sample = []
        absoluteSum = 0
        for j in range(2*k+1):
            pixelValue = allPixelValues[i+j]
            sample.append(pixelValue)
            absoluteSum += abs(pixelValue)
        sample = np.array(sample) / absoluteSum
        #compare sample to greyscale model
        fitMeasure = mahalanobisDistance(sample, modelMean, modelCovarMatrix)
        fitValues.append(fitMeasure)
    #get index of best fitting point (smallest fit value)
    sortedFitIndices = fitValues.argsort()
    indexOfBestPixel = sortedFitIndices[0] + k
    #get pixelcoordinates of best fitting point
    pixelXIndex = 2*indexOfBestPixel
    pixelYIndex = pixelXIndex + 1
    return allPixels[pixelXIndex], allPixels[pixelYIndex]
    
def mahalanobisDistance(sample, mean, covarianceMatrix):
    d = len(sample)
    d1, d2 = covarianceMatrix.shape
    if(len(mean)!=d or d1!=d or d2!=d):
        print 'Dimensions are incorrect!'
    difference = np.matrix(sample - mean)
    result = np.matrix.transpose(difference) * np.linalg.inv(covarianceMatrix) * difference
    return result
    
def buildAllGreyscaleModels(landmarks, nbOfSamplesPerSide, gradientGrayscaleImages):
    #landmarks is a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    if(len(gradientGrayscaleImages)!=landmarks.shape[0]):
        print 'Landmarks do not correspond to images!'
    #get the normals of all landmarks and put them in a structure similar to that of the landmarks
    allLandmarkNormals = calculateAllLandmarkNormals(landmarks)
    allModelCovarMatrices = []
    allModelMeans = []
    #for each tooth
    for i in range(landmarks.shape[1]):
        modelCovarMatricesForTooth = []
        modelMeansForTooth = []
        #for each landmark of this tooth
        for j in range(landmarks.shape[2]/2):
            modelLandmarks = []
            modelLandmarkNormals = []
            xIndex = 2*j
            yIndex = 2*j+1
            #for each image
            for k in range(landmarks.shape[0]):
                landmarkX = landmarks[k][i][xIndex]
                landmarkY = landmarks[k][i][yIndex]
                landmarkNormalX = allLandmarkNormals[k][i][xIndex]
                landmarkNormalY = allLandmarkNormals[k][i][yIndex]
                modelLandmarks.extend([landmarkX,landmarkY])
                modelLandmarkNormals.extend([landmarkNormalX,landmarkNormalY])
            #construct the greyscale model for landmark j in tooth i
            """modelCovarMatrix, modelMean = buildRegularGrayscaleModel(modelLandmarks, modelLandmarkNormals, nbOfSamplesPerSide, gradientGrayscaleImages)"""
            modelCovarMatrix, modelMean = buildDerivativeGrayscaleModel(modelLandmarks, modelLandmarkNormals, nbOfSamplesPerSide, gradientGrayscaleImages)
            modelCovarMatricesForTooth.append(modelCovarMatrix)
            modelMeansForTooth.append(modelMean)
        allModelCovarMatrices.append(modelCovarMatricesForTooth)
        allModelMeans.append(modelMeansForTooth)
    #allModelCovarMatrices contains all model covariance matrices (one for each landmark) and has dimensions tooth X landmark
    #allModelMeans contains all model means (one for each landmark) and has dimensions tooth X landmark
    return allModelCovarMatrices, allModelMeans
    
def calculateNewLandmarksForToothInImage(landmarks, nbOfSamplesPerSide, modelMeans, modelCovarMatrices, gradientGreyscaleImage):
    #landmarks is an array of interleaved x and y coordinates for all landmarks of one tooth in one image
    #modelMeans is an array of model means (one for each landmark)
    #modelCovarMatrices is an array of model covariance matrices (one for each landmark)
    #gradientGreyscaleImage is the image in which the previous variables were evaluated
    landmarkNormals = calculateLandmarkNormals(landmarks)
    newLandmarks = []
    for i in range(len(landmarks)/2):
        """newLandmarkX, newLandmarkY = calculateNewLandmarkWithRegularGrayscaleModel(landmarks[2*i], landmarks[2*i+1], landmarkNormals[2*i], landmarkNormals[2*i+1], nbOfSamplesPerSide, modelMeans[i], modelCovarMatrices[i], gradientGreyscaleImage)"""
        newLandmarkX, newLandmarkY = calculateNewLandmarkWithDerivativeGrayscaleModel(landmarks[2*i], landmarks[2*i+1], landmarkNormals[2*i], landmarkNormals[2*i+1], nbOfSamplesPerSide, modelMeans[i], modelCovarMatrices[i], gradientGreyscaleImage)
        newLandmarks.extend([newLandmarkX,newLandmarkY])
    #newLandmarks is an array of interleaved x and y coordinates for all new landmarks of this tooth in this image
    return newLandmarks
    
if __name__ == '__main__':
    landmarks=prep.load_landmark_data('_Data/Landmarks/original', 14)
    images = prep.import_images('_Data/Radiographs', False)
    prepImages = prep.preprocess_all_images(images, False)
    #prepImages = prep.convertImagesToGrayscale(prepImages, True)
    buildAllGreyscaleModels(landmarks, 5, prepImages)