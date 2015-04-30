import numpy as np

def getLandmarksOfTooth(landmarks, toothNum):
    Outlandmarks = np.zeros((14,80))
    for i in range(14):
        Outlandmarks[i] = landmarks[i][toothNum][:]
    return Outlandmarks