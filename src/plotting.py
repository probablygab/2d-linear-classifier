# AUXILIARY PLOTTING FUNCTIONS

import matplotlib.pyplot as plt
import numpy as np

# Global plot style
plt.style.use('bmh')

def plotPoints(points, color='b', marker='o', scaleEqual=True, label=None):
    '''
    Plot points

    Note
    ----
    points is a list of np.array and each array MUST have shape (2,)
    '''

    # Adjust scale
    if scaleEqual: plt.axis('equal')

    xP = [p[0] for p in points]
    yP = [p[1] for p in points]

    if label == None: plt.scatter(x=xP, y=yP, color=color, marker=marker)
    else:             plt.scatter(x=xP, y=yP, color=color, marker=marker, label=label), plt.legend(loc='upper left')

def plotSegments(segments, color='b', marker='o', linestyle='-', scaleEqual=True):
    '''
    Plot line segments

    Note
    ----
    segments is a list of np.array and each array MUST have shape (2, 2)
    '''

    # Adjust scale
    if scaleEqual: plt.axis('equal')

    for s in segments:
        xVal = [s[0][0], s[1][0]]
        yVal = [s[0][1], s[1][1]]

        plt.plot(xVal, yVal , color=color, marker=marker, linestyle=linestyle)

def plotConvexHull(allPoints, allSegments, color='b', marker='o', linestyle='-', scaleEqual=True, label=None):
    '''
    Plot convex hull, including
    inner points and convex polygon segments

    Note
    ----
    allPoints is a list of np.array and each array MUST have shape (2,)

    allSegments is a list of np.array and each array MUST have shape (2, 2)
    '''

    # Plot both points and segments
    plotPoints(allPoints, color, marker, scaleEqual=scaleEqual, label=label)
    plotSegments(allSegments, color, marker, linestyle, scaleEqual=scaleEqual)

    plt.ylabel("Y axis")
    plt.xlabel("X axis")

def plotLineFromEquation(slope, intercept, xp, yp, color='blue', scaleEqual=True):
    '''
    Plot line given equation and a point.
    Also draw y = ax + b around a point

    Parameters
    ----------
    slope: slope of the line
    intercept: y-intercept of the line
    xp, yp: coordinates of any point from given equation

    Note
    ----
    Slope cannot be zero
    '''

    # Adjust scale
    if scaleEqual: plt.axis('equal')

    plotPoints([np.array([xp, yp])], color=color, scaleEqual=scaleEqual) 

    text = f"y = {slope:.3f}x {intercept:.3f}"

    plt.axline([xp, yp], slope=slope, color=color, label=text)
    plt.legend(loc='upper left')

    plt.ylabel("Y axis")
    plt.xlabel("X axis")