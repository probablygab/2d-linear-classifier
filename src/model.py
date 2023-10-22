# MODEL CREATION / MANIPULATION FUNCTIONS

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from src.plotting import plotSegments, plotConvexHull, plotLineFromEquation

from src.geometry import getOrientation, getDirection, grahamScan, linearSweeping, \
                         findClosestPointsBetweenTwoConvexHulls, findPerpendicularLineEq

from src.geometry import TURN_RIGHT

# AUXILIARY MODEL FUNCTIONS

def checkPointInModel(p, slope, intercept) -> int:
    '''
    Check which class a point is in given a linear model (Ax + B)

    Parameters
    ----------
    p: point to check
    slope: line equation slope -> A
    intercept: line equation intercept -> B

    Returns
    -------
    1: point below line
    -1: point above line
    0: point on the line
    '''
    
    segmentYWhenX0 = slope * 0 + intercept
    segmentYWhenX1 = slope * 1 + intercept

    vectorX0Y = np.array([0, segmentYWhenX0])
    vectorX1Y = np.array([1, segmentYWhenX1])

    return getOrientation(vectorX0Y, p, vectorX1Y)
    
def getOrientationConvexHullTargetToModel(trainingData, trainingDataLabel, slope, intercept, targetToBeSeparated):
    '''
    Check which orientation to check for a point for it to be considered as a part of class target.

    Parameters
    ----------
    trainingData: np.array of points used for training the model
    trainingDataLabel: np.array of labels classifying the trainingData points

    slope: line equation slope -> A
    intercept: line equation intercept -> B
    targetToBeSeparated: target label

    Returns
    -------
    1: target is below line
    -1: target is above line
    0: target is on the line (will not happen)
    '''

    point = 0
    for i in range(len(trainingData)):
        if trainingDataLabel[i] == targetToBeSeparated:
            point = trainingData[i]
            break

    return checkPointInModel(point, slope, intercept)

# MAIN MODEL FUNCTIONS

def plotConvexHullDatabases(databaseTarget, databaseAfterPCA, scaleEqual=True):
    '''
    Plots an arbitrary number of convex hulls defining the database.

    Automatically detects the amount of labels (targets) and assigns colors to them

    Parameters
    ----------
    databaseTarget: np.array of labels classifying the database points
    databaseAfterPCA: np.array of two-dimensional points
    scaleEqual: if set to False, will adjust plot scale to whatever shape the hulls end up to be
    '''

    separateData = {}

    # Count number of targets
    for t in databaseTarget:
        separateData[str(t)] = []

    # Get points of every target
    for i in range(databaseAfterPCA.shape[0]):
        oldValue = separateData[str(databaseTarget[i])]

        oldValue.append(databaseAfterPCA[i])
        separateData[str(databaseTarget[i])] = oldValue

    # Assign colors
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(separateData))))

    # Plot every hull
    for label, points in separateData.items():
        hullSegments, _ = grahamScan(points)

        plotConvexHull(points, hullSegments, marker='.', color=next(color), scaleEqual=scaleEqual, label=str(label))

def generateTrainingAndTestingData(database, databaseTarget):
    '''
    Generates a subset of database for training and testing

    Points are chosen at random

    70% -> training and 30% -> testing

    Parameters
    ----------
    database: np.array of two-dimensional points
    databaseTarget: np.array of labels classifying the database points

    Returns
    -------
    Tuple of shape (trainingData, trainingDataLabel, testData, testDataLabel)
    '''

    database70 = []
    database70target = []
    database30 = []
    database30target = []

    for (i, j) in zip(database, databaseTarget):
        rand = np.random.randint(0, 100)
        if (rand >= 70):
            database30.append(i)
            database30target.append(j)
        else:
            database70.append(i)
            database70target.append(j)
    
    database70       = np.array(database70)
    database70target = np.array(database70target)
    database30       = np.array(database30)
    database30target = np.array(database30target)

    return (database70, database70target, database30, database30target)

def filterTrainingDataToCreateModel(trainingData, trainingDataLabel, testData, testDataLabel, axis, low, high):
    '''
    Filters a section of training data points and adds them to test data

    Parameters
    ----------
    trainingData: np.array of two-dimensional points
    trainingDataLabel: np.array of labels classifying the training database points
    testData: np.array of two-dimensional points
    testDataLabel: np.array of labels classifying the test database points

    axis: axis to filter points
    low: axis section lower bound
    high: axis section upper bound 

    Returns
    -------
    Tuple of shape (trainingData, trainingDataLabel, testData, testDataLabel)
    '''

    trainingDataList      = [x for x in trainingData]
    trainingDataLabelList = [x for x in trainingDataLabel]
    testDataList          = [x for x in testData]
    testDataLabelList     = [x for x in testDataLabel]

    trainingDataListNew = []
    trainingDataLabelListNew = []

    # Filter points from section
    for i in range(len(trainingDataList)):
        if trainingDataList[i][axis] > low and trainingDataList[i][axis] < high:
            testDataList.append(trainingDataList[i])
            testDataLabelList.append(trainingDataLabelList[i])
        else:
            trainingDataListNew.append(trainingDataList[i])
            trainingDataLabelListNew.append(trainingDataLabelList[i])

    # Convert lists back no np.arrays
    trainingData      = np.array(trainingDataListNew)
    trainingDataLabel = np.array(trainingDataLabelListNew)
    testData          = np.array(testDataList)
    testDataLabel     = np.array(testDataLabelList)

    return trainingData, trainingDataLabel, testData, testDataLabel

def generateTrainingAndTestingDataFiltered(database, databaseTarget, axis, low, high):
    '''
    Generates a subset of database for training and testing

    Filters a section of points and adds them to test data

    Points out of the section are chosen at random

    70% -> training and 30% -> testing

    Parameters
    ----------
    database: np.array of two-dimensional points
    databaseTarget: np.array of labels classifying the database points

    axis: axis to filter points
    low: axis section lower bound
    high: axis section upper bound 

    Returns
    -------
    Tuple of shape (trainingData, trainingDataLabel, testData, testDataLabel)
    '''

    trainingData, trainingDataLabel, testData, testDataLabel = generateTrainingAndTestingData(database, databaseTarget)

    return filterTrainingDataToCreateModel(trainingData, trainingDataLabel,
                                           testData, testDataLabel,
                                           axis, low, high)

def separateDatabaseByTarget(database, databaseTarget, targetId):
    '''
    Generates one subset of the database including the target ID and one subset without it

    equalData -> points with the targetId label

    diffData -> points without the targetId label

    Parameters
    ----------
    database: np.array of two-dimensional points
    databaseTarget: np.array of labels classifying the database points
    targetId: target label to split the database

    Returns
    -------
    Tuple of shape (equalData, equalLabelData, diffData, diffLabelData)
    '''

    equal, equalLabel = [], []
    diff, diffLabel = [], []

    # Separate points given target ID
    for (data, label) in zip(database, databaseTarget):
        if (label == targetId):
            equal.append(data)
            equalLabel.append(label)
        else:
            diff.append(data)
            diffLabel.append(label)
    
    # Convert lists to np.array
    equal      = np.array(equal)
    equalLabel = np.array(equalLabel)
    diff       = np.array(diff)
    diffLabel  = np.array(diffLabel)

    return (equal, equalLabel, diff, diffLabel)

def checkInnerHull(hull1: list, hull2: list) -> bool:
    '''
    Checks if a convex hull is completely inside another

    Parameters
    ----------
    hull1: list of points of hull1 each as an np.array of shape (2,)
    hull2: list of points of hull2 each as an np.array of shape (2,)

    Returns
    -------
    True if hull1 is completely inside hull2, false otherwise
    '''

    for j in range(len(hull2) - 1):
        for i in range(len(hull1)):
            if getDirection(hull2[j], hull2[j+1], hull1[i]) == TURN_RIGHT:
                return False

    return True

def generateModelAndPlot(trainingData, trainingDataLabel, targetLabelId, scaleEqual=True):
    '''
    Generates a binary linear model (Ax + B) for a set of points given a targetLabelId

    The resulting model can classify data as FROM targetLabel or NOT FROM targetLabel

    Does not generate a model if the data is not separable for targetLabelId

    If generation is successful, plots the model

    Parameters
    ----------
    trainingData: np.array of two-dimensional points
    trainingDataLabel: np.array of labels classifying the training database points
    targetLabelId: label to be classified
    scaleEqual: if set to False, will adjust plot scale to whatever shape the hulls end up to be

    Returns
    -------
    Tuple of shape (slope, intercept), which defines Ax + B for the model
    '''
    
    # Separate data
    dataLabel, labelTarget, dataDiff, diffTarget = separateDatabaseByTarget(trainingData, trainingDataLabel, targetLabelId)

    # Generate hulls and check intersection
    labelSeg, labelPoints = grahamScan([x for x in dataLabel])
    diffSeg, diffPoints = grahamScan([x for x in dataDiff])

    hullsIntersect = linearSweeping(labelSeg + diffSeg, True)

    # Checks for intersection or a hull inside another
    if hullsIntersect:
        print(f"Selected hulls intersect for target ID {targetLabelId}. Unable to generate model due to data being inseparable")
        return
    elif checkInnerHull(labelPoints, diffPoints) or checkInnerHull(diffPoints, labelPoints):
        print(f"At least one hull is inside another for target ID {targetLabelId}. Unable to generate model due to data being inseparable")
        return
    else:
        print("Selected hulls do not intersect, generating model...")

    # PLOT MODEL

    # Plot hulls
    plotConvexHull([x for x in dataLabel], labelSeg, marker='.', color='green', scaleEqual=scaleEqual)
    plotConvexHull([x for x in dataDiff], diffSeg, marker='.', scaleEqual=scaleEqual)

    # Find equation for model
    closestSeg = findClosestPointsBetweenTwoConvexHulls(labelPoints, diffPoints)

    slope, intercept, xMiddle, yMiddle = findPerpendicularLineEq(closestSeg)

    # Plot model equation
    plotSegments([closestSeg], color='red', scaleEqual=scaleEqual)

    plotLineFromEquation(slope, intercept, xMiddle, yMiddle, color='black', scaleEqual=scaleEqual)

    return slope, intercept

def runModelAndGetMetrics(testData, testDataLabel, targetId, targetIdOrientation, eqSlope, eqIntercept):
    '''
    For every point in testData, check the model prediction and collect results

    Calculates the precision, recall and f1-score of the model

    Parameters
    ----------
    testData: np.array of two-dimensional points
    testDataLabel: np.array of labels classifying the testData points

    targetId: label to be classified
    targetIdOrientation: flag for which side of the line the label data is on

    eqSlope: model line equation slope -> A
    eqIntercept: model line equation intercept -> B

    Returns
    -------
    Tuple of shape (precision, recall, f1)
    '''

    # Model results
    testTarget = []

    # Get model results
    for point in testData:
        if (checkPointInModel(point, eqSlope, eqIntercept) == targetIdOrientation):
            testTarget.append(targetId)
        else:
            testTarget.append(-100) # Not ID

    # Count hits and misses
    TP, FP, TN, FN = 0, 0, 0, 0

    for i,j in zip(testDataLabel, testTarget):
        if (i == j == targetId):
            TP += 1
        elif (i != targetId and j != targetId):
            TN += 1
        elif (i != j):
            if j == targetId:
                FP += 1
            elif j != targetId:
                FN += 1

    # Taken from https://en.wikipedia.org/wiki/Precision_and_recall 
    # and https://en.wikipedia.org/wiki/F-score

    # Get statistics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)

    print("True Positive:", TP, "False Positive:", FP, "True Negative:", TN, "False Negative:", FN, end="\n\n")

    return (precision * 100, recall * 100, f1 * 100)

# AUX FUNCTION TO MAKE PANDAS DF FROM CSV COMPATIBLE WITH ALL OTHER FUNCTIONS

def getNumericTargetFromDataframe(dfColumn):
    '''
    Converts label strings to unique numbers

    Parameters
    ----------
    dfColumn: column of the dataframe with label strings

    Returns
    -------
    List with string values mapped to integers
    '''

    targetDict = {}
    count = 0

    numTarget = []

    for row in dfColumn:
        if targetDict.__contains__(str(row)):
            numTarget.append(targetDict[str(row)])
        else:
            targetDict[str(row)] = count
            count += 1
            numTarget.append(targetDict[str(row)])
    
    return numTarget