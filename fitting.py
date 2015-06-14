import cv2
import numpy as np
import math
import importingAndPreprocessing as prep
import tests
import alignment
import modeling
import tools
    
def matchModelToShapeIteration(meanShapeLandmarks, P, principalEigenvalues, shapeLandmarks, landmarkWeights, b):
    dot = np.dot(P,b)
    temp = np.zeros([dot.shape[0]])
    for i in range(dot.shape[0]):
        temp[i] = dot[i]
    dot = np.array(temp)
    
    modelLandmarks = meanShapeLandmarks + dot
    transformation = alignment.alignFirstToSecondTooth(modelLandmarks, shapeLandmarks, landmarkWeights)
    transformedModelLandmarks = alignment.transformLandmarks(transformation, modelLandmarks)
    #invTransformation = alignment.invertTransformation(transformation)
    invTransformation = alignment.alignFirstToSecondTooth(shapeLandmarks, modelLandmarks, landmarkWeights)
    transformedShapeLandmarks = alignment.transformLandmarks(invTransformation, shapeLandmarks)
    intermediateDot = np.dot(transformedShapeLandmarks, meanShapeLandmarks)
    #calculate a projection matrix for the orthogonal projection onto the plane tangent to the mean
    orthogonalProjectionMatrix = np.array(np.zeros([len(meanShapeLandmarks), len(meanShapeLandmarks)]))
    np.fill_diagonal(orthogonalProjectionMatrix, 1)
    outerMeanProduct = np.array(np.outer(meanShapeLandmarks, meanShapeLandmarks))
    innerMeanProduct = np.dot(meanShapeLandmarks, meanShapeLandmarks)
    quotient = np.array(outerMeanProduct) / innerMeanProduct
    orthogonalProjectionMatrix = orthogonalProjectionMatrix - quotient
    transformedShapeLandmarksInTangentPlane = np.array(transformedShapeLandmarks)# / intermediateDot
    #transformedShapeLandmarksInTangentPlane = np.dot(orthogonalProjectionMatrix, np.array(transformedShapeLandmarksInTangentPlane))
    toMultiplyWithP = np.array(transformedShapeLandmarksInTangentPlane - meanShapeLandmarks)# / intermediateDot
    #toMultiplyWithP = np.dot(orthogonalProjectionMatrix, np.array(toMultiplyWithP))
    newB = np.dot(np.transpose(P), toMultiplyWithP)
    newB = adaptToModelBoundaries(newB, principalEigenvalues)
    return transformation, newB, transformedModelLandmarks
    
def matchModelToShapeIterationConvergenceCheck(oldB, newB, oldTransformation, newTransformation):
    converged = tools.valuesConvergenceCheck(oldB, newB) and tools.valuesConvergenceCheck(oldTransformation, newTransformation)
    return converged
    
def adaptToModelBoundaries(b, principalEigenvalues):
    maxValue = np.max(principalEigenvalues)
    adaptedB = np.zeros(b.shape)
    for i in range(len(b)):
        #rightBoundary = 3*math.sqrt(principalEigenvalues[i])
        rightBoundary = 3*math.sqrt(maxValue)
        leftBoundary = -rightBoundary
        if (b[i] < leftBoundary):
            adaptedB[i] = leftBoundary
        elif (b[i] > rightBoundary):
            adaptedB[i] = rightBoundary
        else:
            adaptedB[i] = b[i]
    return adaptedB
        
def matchModelToShape(meanShapeLandmarks, principalEigenvalues, principalEigenvectors, shapeLandmarks, landmarkWeights):    
    principalEigenvectors = np.transpose(np.array(principalEigenvectors))
    prevB = np.transpose(np.matrix(np.zeros([len(principalEigenvalues)])))
    prevTransformation = (1,0,0,0)
        
    temp = np.zeros([meanShapeLandmarks.shape[0]])
    for i in range(meanShapeLandmarks.shape[0]):
        temp[i] = meanShapeLandmarks[i]
    meanShapeLandmarks = np.array(temp)
    
    converged = False
    while converged is False:
        nextTransformation, nextB, modelLandmarks = matchModelToShapeIteration(meanShapeLandmarks, principalEigenvectors, principalEigenvalues, shapeLandmarks, landmarkWeights, prevB)
        converged = matchModelToShapeIterationConvergenceCheck(prevB, nextB, prevTransformation, nextTransformation)
        prevB = nextB
        prevTransformation = nextTransformation
    return nextTransformation, nextB, modelLandmarks
    
if __name__ == '__main__':
    landmarks = prep.load_landmark_data('_Data/Landmarks/original', 14)
    images = prep.import_images('_Data/Radiographs', False)
    aligned = alignment.alignment(landmarks)
    toothNb = 0
    imageNb = 8
    alignedLandmarks = aligned[3]
    alignedLandmarksForTooth = tools.getLandmarksOfTooth(alignedLandmarks, toothNb)
    model = modeling.PCA(alignedLandmarksForTooth, cutOffValue=None, nbOfComponents=40)#cutOffValue=(100-10e-10), nbOfComponents=None)
    #models = modeling.allPCA(aligned[3], cutOffValue=99.999999, nbOfComponents=None)
    if True:
    #for toothNb in range(len(models)):
        #model = models[toothNb]
        print 'sqrt of variances'
        print [math.sqrt(x) for x in model[1]]
        landmarkWeights = alignment.calculateLandmarkWeightsForAllTeeth(landmarks)
        t, b, transformedLandmarks = matchModelToShape(model[0], model[1], model[2], landmarks[imageNb][toothNb], landmarkWeights[toothNb])
        print 'b'
        print b
        tests.show_landmarks_one_tooth_on_image_dynamic(images[imageNb], landmarks[imageNb][toothNb],'original landmarks for tooth '+str(toothNb)+' on image '+str(imageNb))
        tests.show_landmarks_one_tooth_on_image_dynamic(images[imageNb], transformedLandmarks,'transformed model landmarks for tooth '+str(toothNb)+' on image '+str(imageNb))
        t = alignment.alignFirstToSecondTooth(model[0], landmarks[imageNb][toothNb], landmarkWeights[toothNb])
        transformedMean = alignment.transformLandmarks(t, model[0])
        tests.show_landmarks_one_tooth_on_image_dynamic(images[imageNb], transformedMean,'mean landmarks for tooth '+str(toothNb)+' on image '+str(imageNb))
        cv2.destroyAllWindows()