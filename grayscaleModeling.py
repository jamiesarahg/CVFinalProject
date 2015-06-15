import cv2
import numpy as np
import math
import importingAndPreprocessing as prep
import tests
import alignment
import modeling
import tools
import copy
import searching

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

def plotLandmarksAndNormals(landmarks, landmarkNormals, normalScale, imgs, show=False):
    images = copy.deepcopy(imgs)
    #for each image
    for k in range(landmarks.shape[0]):
        #for each tooth
        for i in range(landmarks.shape[1]):
            #for each landmark of this tooth
            for j in range(landmarks.shape[2]/2):
                xLandmark = landmarks[k][i][2*j]
                yLandmark = landmarks[k][i][2*j+1]
                xNormal = landmarkNormals[k][i][2*j]
                yNormal = landmarkNormals[k][i][2*j+1]
                images[k] = plotLandmarkAndNormal(xLandmark, yLandmark, xNormal, yNormal, normalScale, images[k], False)
        if show:
            cv2.imshow('plottedLandmarksAndNormals',cv2.resize(images[k], (0,0), fx=0.5, fy=0.5))
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    return images
    
def plotLandmarkAndNormal(xLandmark, yLandmark, xNormal, yNormal, normalScale, img, show=False):
    image = copy.deepcopy(img)
    cv2.circle(image,(int(xLandmark),int(yLandmark)),1,cv2.cv.CV_RGB(255, 255, 255),2, 8, 0 )
    nx = xLandmark + normalScale*xNormal
    ny = yLandmark + normalScale*yNormal
    cv2.line(image, (int(xLandmark),int(yLandmark)), (int(nx),int(ny)), cv2.cv.CV_RGB(255, 255, 255), 1)
    if show:
        cv2.imshow('plottedLandmarkAndNormal',cv2.resize(image, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
    return image
    
def plotPixels(pixels, img, show=False):
    image = copy.deepcopy(img)
    for i in range(len(pixels)/2):
        cv2.circle(image,(int(pixels[2*i]),int(pixels[2*i+1])),1,cv2.cv.CV_RGB(255, 255, 255),2, 8, 0 )
    if show:        
        cv2.imshow('plottedPixels',cv2.resize(image, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
    return image

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
    if(crossProductZ(prevVectorX, prevVectorY, n1x, n1y) > 0):
        n1x = -n1x
        n1y = -n1y
    #normalize this surface normal
    n1x, n1y = normalizeVector(n1x, n1y)
    #calculate the second surface vector
    nextVectorX = nextX-centerX
    nextVectorY = nextY-centerY
    #calculate the second surface normal
    n2x, n2y = calculatePerpendicularVector(nextVectorX, nextVectorY)
    #make sure this surface normal is pointing outwards
    if(crossProductZ(nextVectorX, nextVectorY, n2x, n2y) < 0):
        n2x = -n2x
        n2y = -n2y
    #normalize this surface normal
    n2x, n2y = normalizeVector(n2x, n2y)
    #the required landmark normal is just the sum of the two calculated surface normals
    normalX = n1x + n2x
    normalY = n1y + n2y
    normalX, normalY = normalizeVector(normalX, normalY)
    return normalX, normalY
    
def calculatePerpendicularVector(vectorX, vectorY):
    normalX = -vectorY
    normalY = vectorX
    return normalX, normalY
    
def normalizeVector(vectorX, vectorY):
    length = lengthOfVector(vectorX, vectorY)
    normalizedX = vectorX/length
    normalizedY = vectorY/length
    return normalizedX, normalizedY
    
def lengthOfVector(vectorX, vectorY):
    length = math.sqrt((vectorX*vectorX) + (vectorY*vectorY))
    return length
    
def crossProductZ(vector1X, vector1Y, vector2X, vector2Y):
    z = (vector1X*vector2Y) - (vector1Y*vector2X)
    return z
    
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
        pixelX = int(round(points[2*i]))
        pixelY = int(round(points[2*i+1]))
        pixels.extend([pixelX, pixelY])
    return pixels
    
def calculateDerivatives(pixelValues):
    derivativeValues = []
    for i in range(len(pixelValues)-1):
        if(pixelValues[i+1] >= pixelValues[i]):
            derivative = float(pixelValues[i+1] - pixelValues[i])
        else:
            derivative = -float((pixelValues[i] - pixelValues[i+1]))
        derivativeValues.append(derivative)
    return derivativeValues
    
def normalizeValues(vals):
    values = copy.deepcopy(vals)
    total = absoluteSum(values)
    if (total == 0):
        total = 1
    newValues = []
    for value in values:
        newValue = value / total
        newValues.append(newValue)
    return newValues
    
def absoluteSum(values):
    total = 0
    for value in values:
        total += abs(value)
    return total
    
def buildDerivativeGrayscaleModel(landmarks, landmarkNormals, nbOfSamplesPerSide, grayscaleImages, imgsToWriteOn=None):
    #grayscaleImages is an array of all the images, after being converted into a gradient grayscale, in the training set
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
        img = grayscaleImages[i]
        for j in range(len(allPixels[0])/2):
            pixelX = allPixels[i][2*j]
            pixelY = allPixels[i][2*j+1]
            pixelValue = img[pixelY][pixelX]
            pixelValues.append(pixelValue)
        if imgsToWriteOn is not None:
            imgsToWriteOn[i] = plotPixels(allPixels[i], imgsToWriteOn[i])
        derivativeValues = calculateDerivatives(pixelValues)
        derivativeValues = normalizeValues(derivativeValues)
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
    return covarianceMatrix, meanDerivativeValues, imgsToWriteOn
    
def buildAllGreyscaleModels(landmarks, nbOfSamplesPerSide, grayscaleImages, showPoints=False):
    imgsWithSamples = copy.deepcopy(grayscaleImages)
    #landmarks is a three dimentional array of the images, each with arrays for the eight teeth, each with the landmark data
    if(len(grayscaleImages)!=landmarks.shape[0]):
        print 'Landmarks do not correspond to images!'
    #get the normals of all landmarks and put them in a structure similar to that of the landmarks
    allLandmarkNormals = calculateAllLandmarkNormals(landmarks)
    allModelCovarMatrices = []
    allModelMeans = []
    #for each tooth
    for i in range(landmarks.shape[1]):
        modelCovarMatricesForTooth = []
        modelMeansForTooth = []
        #print ' '
        #print 'tooth '+str(i+1)+'/'+str(landmarks.shape[1])
        #print ' '
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
            #print 'model '+str(j+1)+'/'+str(landmarks.shape[2]/2)
            modelCovarMatrix, modelMean, imgsWithSamples = buildDerivativeGrayscaleModel(modelLandmarks, modelLandmarkNormals, nbOfSamplesPerSide, grayscaleImages, imgsWithSamples)
            modelCovarMatricesForTooth.append(modelCovarMatrix)
            modelMeansForTooth.append(modelMean)
        allModelCovarMatrices.append(modelCovarMatricesForTooth)
        allModelMeans.append(modelMeansForTooth)
    if showPoints:
        plotLandmarksAndNormals(landmarks, allLandmarkNormals, 10, grayscaleImages, True)
        for img in imgsWithSamples:
            cv2.imshow('plottedSamplesOnNormals',cv2.resize(img, (0,0), fx=0.5, fy=0.5))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    #allModelCovarMatrices is a two dimentional array of the teeth, each with the landmark covariance matrices (across all images)
    #allModelMeans is a two dimentional array of the teeth, each with the landmark means (across all images)
    return allModelCovarMatrices, allModelMeans
    
def calculateNewLandmarkWithDerivativeGrayscaleModel(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, nbOfSamplesPerSide, modelMean, modelCovarMatrix, grayscaleImage):
    m = nbOfSamplesPerSide
    k = len(modelMean)/2
    if(m <= k):
        print 'M is not larger than k!!'
    #retrieve the pixelcoordinates for all the necessary pixels on the landmark normal (= 2*m + 1 pixels)
    points = pointsOnLandmarkNormal(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, m)
    allPixels = pointsToPixels(points)
    allPixelValues = []
    #retrieve the pixelvalues for all pixels
    for i in range(len(allPixels)/2):
        pixelX = allPixels[2*i]
        pixelY = allPixels[2*i+1]
        pixelValue = grayscaleImage[pixelY][pixelX]
        allPixelValues.append(pixelValue)
    #retrieve the derivatives for all pixelvalues
    allDerivativeValues = calculateDerivatives(allPixelValues)
    #debugging(landmarkX, landmarkY, landmarkNormalX, landmarkNormalY, nbOfSamplesPerSide, grayscaleImage)
    #print allDerivativeValues
    #construct 2*(m-k)+1 samples to compare to the greyscale model
    fitValues = np.zeros([2*(m-k)+1])
    for i in range(2*(m-k)+1):
        sampleDerivatives = allDerivativeValues[i:(i+2*k)]
        sampleDerivatives = normalizeValues(sampleDerivatives)
        #compare sample to greyscale model
        fitMeasure = mahalanobisDistance(sampleDerivatives, modelMean, modelCovarMatrix)
        fitValues[i] = fitMeasure
    #get index of best fitting point (smallest fit value)
    sortedFitIndices = fitValues.argsort()
    indexOfBestPixel = sortedFitIndices[0] + k
    #get pixelcoordinates of best fitting point
    pixelXIndex = 2*indexOfBestPixel
    pixelYIndex = pixelXIndex + 1
    return allPixels[pixelXIndex], allPixels[pixelYIndex]
    
def debugging(xLandmark, yLandmark, xNormal, yNormal, normalScale, img):
    image = copy.deepcopy(img)
    cv2.circle(image,(int(xLandmark),int(yLandmark)),1,cv2.cv.CV_RGB(255, 255, 255),2, 8, 0 )
    nx1 = xLandmark + normalScale*xNormal
    ny1 = yLandmark + normalScale*yNormal
    nx2 = xLandmark - normalScale*xNormal
    ny2 = yLandmark - normalScale*yNormal
    cv2.line(image, (int(nx1),int(ny1)), (int(nx2),int(ny2)), cv2.cv.CV_RGB(255, 255, 255), 1)
    cv2.imshow('plottedLandmarkAndNormal',cv2.resize(image, (0,0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def mahalanobisDistance(sample, mean, covarianceMatrix):
    d = len(sample)
    d1, d2 = covarianceMatrix.shape
    if(len(mean)!=d or d1!=d or d2!=d):
        print 'Dimensions are incorrect!'
    difference = np.matrix(sample - mean)
    try:
        invCovarMatrix = np.linalg.inv(covarianceMatrix)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            invCovarMatrix = np.linalg.pinv(covarianceMatrix)
        else:
            raise
    result = np.dot(difference, invCovarMatrix)
    result = np.dot(np.array(result), np.transpose(difference))
    return result
    
def calculateNewLandmarksForToothInImage(landmarks, nbOfSamplesPerSide, grayscaleModelMeans, grayscaleModelCovarMatrices, gradientGreyscaleImage):
    #landmarks is an array of interleaved x and y coordinates for all landmarks of one tooth in one image
    #grayscaleModelMeans is an array of model means (one for each landmark)
    #grayscaleModelCovarMatrices is an array of model covariance matrices (one for each landmark)
    #gradientGreyscaleImage is the image in which the previous variables were evaluated
    landmarkNormals = calculateLandmarkNormals(landmarks)
    newLandmarks = []
    for i in range(len(landmarks)/2):
        newLandmarkX, newLandmarkY = calculateNewLandmarkWithDerivativeGrayscaleModel(landmarks[2*i], landmarks[2*i+1], landmarkNormals[2*i], landmarkNormals[2*i+1], nbOfSamplesPerSide, grayscaleModelMeans[i], grayscaleModelCovarMatrices[i], gradientGreyscaleImage)
        newLandmarks.extend([newLandmarkX,newLandmarkY])
    #newLandmarks is an array of interleaved x and y coordinates for all new landmarks of this tooth in this image
    return newLandmarks
    
if __name__ == '__main__':
    '''landmarks = prep.load_landmark_data('_Data/Landmarks/original', 14)
    images = prep.import_images('_Data/Radiographs', False, 0)
    prepImages = prep.preprocess_all_images(images, False)
    #prepImages = prep.convertImagesToGrayscale(prepImages, True)
    buildAllGreyscaleModels(landmarks, 10, prepImages, showPoints=False)'''
    markedImages = searching.searchForTeethInImages(20, 40, None, 40)
    for img in markedImages:
        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('search results',small)
        cv2.waitKey(0)
    cv2.destroyAllWindows()