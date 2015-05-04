import numpy as np
import math

def getLandmarksOfTooth(landmarks, toothNum):
    Outlandmarks = np.zeros((14,80))
    for i in range(14):
        Outlandmarks[i] = landmarks[i][toothNum][:]
    return Outlandmarks

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)) #calculate the distances of two landmarks in the sample
    