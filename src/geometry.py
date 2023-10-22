# BASIC GEOMETRY FUNCTIONS

import numpy as np
from functools import cmp_to_key

# Local imports
from src.utils import quickSort, BST

# Constants for readability
TURN_LEFT  = ANTI_CLOCKWISE = -1
TURN_RIGHT = CLOCKWISE      = 1
NO_CHANGE  = COLINEAR       = 0
X_AXIS     = 0
Y_AXIS     = 1

def getOrientation(origin, a, b):
    '''
    Get orientation between two vectors in 2D space

    Parameters
    ----------
    origin: common point between vectors
    a: vector a
    b: vector b

    Returns
    -------
    1: a is clockwise of b
    -1: a is anti-clockwise of b
    0: a and b are colinear
    '''

    # Offset common origin from a and b
    # This way a and b are vectors with origin at (0, 0)
    A = a - origin
    B = b - origin

    return np.sign(A[0] * B[1] - (A[1] * B[0]))

def getDirection(p0, p1, p2):
    '''
    Get turn direction between three consecutive points

    Parameters
    ----------
    p0, p1, p2: consecutive points as in p0 -> p1 -> p2

    Returns
    -------
    1: going from p0 to p2 makes a right turn
    -1: going from p0 to p2 makes a left turn
    0: there is no change of direction going from p0 to p2

    Note
    ----
    Points must have x and y coordinates
    '''

    return getOrientation(p0, p2, p1)

# BASIC LINE GEOMETRY

def isOnSegment(start, end, p) -> bool: 
    '''
    Check if point p is on segment [start <-> end]

    Parameters
    ----------
    start, end: endpoints of segment, each as an np.array of shape (2,)
    p: point in question as an np.array of shape (2,)

    Returns
    -------
    True - p is on segment [start, end]
    False - p is not on segment [start, end]

    Note
    ----
    Points must have x and y coordinates
    '''

    pIsBetweenStartEnd_ifAscending = p[1] >= start[1] and p[1] <= end[1] and p[0] >= start[0] and p[0] <= end[0]
    pIsBetweenStartEnd_ifDescending = p[1] <= start[1] and p[1] >= end[1] and p[0] <= start[0] and p[0] >= end[0]
    
    if (pIsBetweenStartEnd_ifAscending or pIsBetweenStartEnd_ifDescending):
        return True

    return False


def hasIntersection(A, B, ignoreSamePoint=False, epsilon=1e-6) -> bool:
    '''
    Check intersection between two line segments

    Parameters
    ----------
    A, B: segments A and B each as an np.array of shape (2, 2)
    ignoreSamePoint: if set to True, will ignore intersections of overlapping points
    epsilon: tolerance for comparison

    Returns
    -------
    True - There is an intersection
    False - There is not

    Note
    ----
    Segments must have 2 points with x and y coordinates each
    '''

    Astart, Aend, Bstart, Bend = A[0], A[1], B[0], B[1]

    # Edge case for convex hull, ignore intersections when endpoints are roughly the same
    if ignoreSamePoint:
        SS = np.allclose(Astart, Bstart, atol=epsilon, rtol=epsilon)
        SE = np.allclose(Astart, Bend,   atol=epsilon, rtol=epsilon)
        ES = np.allclose(Aend,   Bstart, atol=epsilon, rtol=epsilon)
        EE = np.allclose(Aend,   Bend,   atol=epsilon, rtol=epsilon)

        if SS or SE or ES or EE:
            return False

    # Vanilla steps
    direction1 = getDirection(Astart, Aend, Bstart)
    direction2 = getDirection(Astart, Aend, Bend)
    direction3 = getDirection(Bstart, Bend, Astart)
    direction4 = getDirection(Bstart, Bend, Aend)

    BisBetweenA = (direction3 > 0 and direction4 < 0) or (direction3 < 0 and direction4 > 0)
    AisBetweenB = (direction1 > 0 and direction2 < 0) or (direction1 < 0 and direction2 > 0)

    if (BisBetweenA and AisBetweenB):
        return True
    elif (direction1 == 0):
        return isOnSegment(Astart, Aend, Bstart)
    elif (direction2 == 0):
        return isOnSegment(Astart, Aend, Bend)
    elif (direction3 == 0):
        return isOnSegment(Bstart, Bend, Astart)
    elif (direction4 == 0):
        return isOnSegment(Bstart, Bend, Aend)
    
    return False

# GRAHAM SCAN IMPLEMENTATION

# Global for anchor value used by compare
global CURRENT_ANCHOR

def grahamCompare(a, b) -> int:
    '''
    Compares two points in relation to an anchor

    Please set a point to CURRENT_ANCHOR global variable before use

    Used in graham scan to determine the anti-clockwise order of points

    Parameters
    ----------
    a: point a
    b: point b

    Returns
    -------
    1: a is further from the anchor than b
    -1: a is closer to the anchor than b
    0: a and b are the same point
    '''

    ori = getOrientation(CURRENT_ANCHOR, a, b)

    # A is "smaller" than B, closer to anchor
    if   ori == CLOCKWISE:
        return -1
    elif ori == COLINEAR:
        # If colinear, choose furthest point
        distA = np.linalg.norm(CURRENT_ANCHOR - a)
        distB = np.linalg.norm(CURRENT_ANCHOR - b)

        if distA > distB:
            return -1 # A is furthest
        elif distA == distB:
            return 0  # Equal distance
        else:
            return 1  # B is furthest 
    
    # A is "greater" than B, further from anchor
    else:    
        return 1
    
def toSegments(hull: list) -> list:
    '''
    Converts hull points to segments, automatically adds last segment

    Parameters
    ----------
    hull: list of hull points

    Returns
    -------
    A list of np.arrays of shape (2, 2), representing each point of a segment
    '''

    segList = []

    for i in range(len(hull) - 1):
        p0 = hull[i]
        p1 = hull[i+1]

        segment = np.array([[p0[0], p0[1]], [p1[0], p1[1]]])
        segList.append(segment)

    # Connect end to start
    p0 = hull[-1]
    p1 = hull[0]

    segment = np.array([[p0[0], p0[1]], [p1[0], p1[1]]])
    segList.append(segment)

    return segList

def getAnchorIdx(array: list, epsilon=1e-6) -> int:
    '''
    Get anchor index for the given array of points

    Anchor is defined as the southmost and leftmost point

    Parameters
    ----------
    array: list of points each as an np.array of shape (2,)
    epsilon: comparison tolerance

    Returns
    -------
    Array index for the anchor point
    '''

    anchorIdx = 0

    # Find southwest-most point
    for i in range(len(array)):
        # If there's a tie, choose point with smallest X
        if abs(array[i][1] - array[anchorIdx][1]) < epsilon:
            if (array[i][0] < array[anchorIdx][0]):
                anchorIdx = i

        # Look for point with smallest Y
        elif array[i][1] < array[anchorIdx][1]:
            anchorIdx = i
    
    return anchorIdx

def grahamScan(array: list) -> (list, list):
    '''
    Generates a convex hull for a set of points

    Parameters
    ----------
    array: list of points each as an np.array of shape (2,)

    Returns
    -------
    Tuple of shape (list of convex hull segments, list of convex hull points)
    '''

    size = len(array)

    # Check for viability
    if size < 3:
        print("Unable to generate convex hull for 2 points or less")
        raise InterruptedError
    
    # Find anchor
    anchorIdx = getAnchorIdx(array)
    
    # Put anchor at the start
    array[0], array[anchorIdx] = array[anchorIdx], array[0]

    # Set anchor used for compare
    global CURRENT_ANCHOR; CURRENT_ANCHOR = array[0]
    
    # Sort in relation to anchor
    quickSort(array, grahamCompare, 1, size - 1)
    #array[1:].sort(key=cmp_to_key(compare)) # Does not work, key function gets wild
    
    # Init points (first 3 always included)
    hull = [array[0], array[1], array[2]]

    # Skip if there's only three points
    if (size > 3):
        for i in range(3, size):
            # Skip equal points
            if np.allclose(hull[-1], array[i]): 
                continue
            
            # Test right turn
            while (getDirection(hull[-2], hull[-1], array[i]) == TURN_RIGHT):
                hull.pop()

            hull.append(array[i]) 

    # Convert hull points to segments
    # Adding points for consecutive segments and connecting last to start
    return toSegments(hull), hull

# LINEAR SWEEPING IMPLEMENTATION

class Endpoint:
    '''
    Class which holds information needed to compare and scan endpoints

    Fields
    ----------
    p: point as an np.array of shape (2,)
    pos: endpoint position in its segment: 'l' for left, 'r' for right
    seg: segment associated with point p
    '''

    def __init__(self, p, pos, seg):
        self.p = p
        self.pos = pos
        self.seg = seg

# Transform segments for comparison
def segmentsToEndpoints(segments: list) -> list:
    '''
    Converts a list of segments to a list of endpoints of those segments

    Parameters
    ----------
    segments: list of segments each as an np.array of shape (2, 2)

    Returns
    -------
    List of endpoints (double the size of the original list)
    '''

    endpoints = []

    for seg in segments:
        # First point is to the left 
        if (seg[0][0] < seg[1][0]):
            endpoints.append(Endpoint(seg[0], 'l', seg))
            endpoints.append(Endpoint(seg[1], 'r', seg))
        # First point is to the right 
        else:
            endpoints.append(Endpoint(seg[0], 'r', seg))
            endpoints.append(Endpoint(seg[1], 'l', seg))
    
    return endpoints

# Compare segments for binary tree
def segmentCompare(a, b, epsilon=1e-6) -> int:
    '''
    Compares segments in relation to one another by their Y coordinate

    Parameters
    ----------
    a: segment a
    b: segment b
    epislon: tolerance for comparison

    Returns
    -------
    1: Segment a is above b
    -1: Segment a is below b
    0: Segments coincide
    '''

    # First y coordinate
    diff = a[0][1] - b[0][1]

    if abs(diff) > epsilon:
        return np.sign(diff)
    
    # Second y coordinate
    diff = a[1][1] - b[1][1]

    if abs(diff) > epsilon:
        return np.sign(diff)

    return 0
    
# Compare endpoints for sorting
def endpointCompare(a, b, epsilon=1e-6) -> int:
    '''
    Compares endpoints in relation to one another by their X coordinate and other stuff

    Parameters
    ----------
    a: endpoint a
    b: endpoint b
    epislon: tolerance for comparison

    Returns
    -------
    1: Endpoint a smaller than b
    -1: Endpoint a is greater than b
    0: Endpoints coincide
    '''

    # Order by x coordinate of point
    diff = a.p[0] - b.p[0]

    # Different x coordinate:
    # Leftmost points first
    if abs(diff) > epsilon:
        return np.sign(diff)
    
    # 1st tiebreaker: Same x coordinate
    # Left endpoints first
    # Naturally 'l' - 'r' gives a negative number
    diff = ord(a.pos) - ord(b.pos)
    
    if diff != 0:
        return diff

    # 2nd tiebreaker: Same x coordinate and same position
    # Lower y coordinates first
    diff = a.p[1] - b.p[1]

    if abs(diff) > epsilon:
        return np.sign(diff)

    # Else they are equal (unlikely)
    return 0

def linearSweeping(segments: list, ignoreSamePoint=False) -> bool:
    '''
    Performs a linear scan across segments checking for intersections

    Parameters
    ----------
    segments: list of segments each as an np.array of shape (2, 2)
    ignoreSamePoint: if set to True, will ignore intersections of overlapping points

    Returns
    -------
    True: at least one segment intersects with another
    False: there is no intersection between segments
    '''

    # Forward flag for hasIntersection
    flag = ignoreSamePoint

    # Get points and position (left or right)
    endpoints = segmentsToEndpoints(segments)

    # Sort endpoints  
    endpoints.sort(key=cmp_to_key(endpointCompare))

    # Init tree
    tree = BST(cmp=segmentCompare)

    # Main loop
    for e in endpoints:
        position, S = e.pos, e.seg

        # Left endpoint
        if position == 'l':
            tree.insert(S)

            B, A = tree.getBelowAbove(S)

            if (A != None and hasIntersection(A.key, S, flag)) \
            or (B != None and hasIntersection(B.key, S, flag)):
                return True
           
        # Right endpoint
        if position == 'r':
            B, A = tree.getBelowAbove(S)

            if (A != None and B != None and hasIntersection(A.key, B.key, flag)):
                return True

            tree.delete(S)

    return False

# FIND PERPENDICULAR LINE IMPLEMENTATION

def findPerpendicularLineEq(segment: np.array):
    '''
    Finds the perpendicular line equation passing through the middle of a segment

    Parameters
    ----------
    segment: segment in question

    Returns
    -------
    Tuple of shape (slope, intercept, xMiddle, yMiddle)

    slope and intercept are real numbers
    
    xMiddle and yMiddle are np.array points of shape (2,)
    '''

    # Find segment line equation
    p0 = segment[0]
    p1 = segment[1]

    slope = (p0[1] - p1[1]) / (p0[0] - p1[0])
    intercept = p0[1] - slope * p0[0]

    # Calculate cartesian distance
    w = p0 - p1

    # Find middle point
    xMiddle = p1[0] + w[0] * 0.5
    yMiddle = p1[1] + w[1] * 0.5

    # New slope for perpendicular
    slope = -1 / slope
    
    # New intercept for perpendicular
    intercept = yMiddle - slope * xMiddle 

    return slope, intercept, xMiddle, yMiddle

# FIND CLOSEST POINTS IMPLEMENTATION

def findClosestPointsBetweenTwoConvexHulls(hull1: list, hull2: list):
    '''
    Finds closest points between two convex hulls.

    Parameters
    ----------
    hull1: First hull points as a list of np.array points of shape (2,)
    hull2: Second hull points as a list of np.array points of shape (2,)

    Returns
    -------
    Segment between the closest points as an np.array of shape (2, 2)
    '''

    smallestDist = np.inf

    for i in hull1:
        for j in hull2:
            distA = np.linalg.norm(i - j)

            if distA < smallestDist:
                smallestDist = distA
                closestPointToHull1 = j
                closestPointToHull2 = i 
                
    return np.array([closestPointToHull2, closestPointToHull1])