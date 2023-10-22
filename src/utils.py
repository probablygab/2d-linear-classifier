# AUXILIARY DATA STRUCTURES AND FUNCTIONS

import copy as cp
import random as rnd

# ======================================================= #
class Stack:
    '''
    Class simulating a stack, used for the iterative quick sort
    '''

    def __init__(self):
        self.size = 0
        self.stack = []

    def push(self, obj):
        self.stack.append(obj)
        self.size += 1

    def pop(self):
        self.stack.pop()
        self.size -= 1

    def top(self):
        return self.stack[-1]

    def toList(self):
        return self.stack
    
    def len(self):
        return self.size
    
    def isEmpty(self):
        return self.size == 0

# ======================================================= #
def selectionSort(array, cmp, start, end):
    '''
    Sort an array in-place.

    Parameters
    ----------
    array: array to sort
    cmp: compare function
    start, end: start and end indexes, both inclusive

    Returns
    -------
    Sorted array
    '''

    for i in range(start, end):
        min = i
        
        for j in range(i + 1, end + 1):  
            if cmp(array[j], array[min]) == -1:
                min = j

        # Avoid swapping same element
        if min != i:
            array[i], array[min] = array[min], array[i]

    return array

# Threshold for using selection sort instead of quick sort
QS_THRESHOLD = 100

def quickSort(array, cmp, start, end):
    '''
    Sort an array in-place. Iterative implementation.
    Uses Selection Sort below a threshold for efficiency.

    Parameters
    ----------
    array: array to sort
    cmp: compare function
    start, end: start and end indexes, both inclusive

    Returns
    -------
    Sorted array
    '''

    # Iterative quicksort
    stack = Stack()
    stack.push((start, end))

    while stack.isEmpty() == False:
        # Get current partition
        (pStart, pEnd) = stack.top()
        stack.pop()

        # Use selection sort when below threshold
        if pEnd - pStart < QS_THRESHOLD:
            selectionSort(array, cmp, pStart, pEnd)
            continue
        
        i, j = pStart, pEnd
        pivot = cp.deepcopy(array[rnd.randint(pStart, pEnd)])

        # Main loop
        while i <= j:
            while cmp(pivot, array[i]) == 1:
                i += 1

            while cmp(pivot, array[j]) == -1:
                j -= 1

            # Swap [i] and [j]
            if i <= j:
                if i != j:
                    array[i], array[j] = array[j], array[i]

                i += 1
                j -= 1
        
        # Left partition
        if pStart < j:
            stack.push((pStart, j))

        # Right partition
        if i < pEnd:
            stack.push((i, pEnd))

    # Sorting happened in-place, but return anyways
    return array

# ======================================================= #
# BINARY TREE IMPLEMENTATION, SOURCE: 
# https://www.geeksforgeeks.org/deletion-in-binary-search-tree/

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self, cmp):
        self.cmp = cmp
        self.root = None

    def inOrder(self):
        inOrderAUX(self.root)

    def insert(self, key):
        self.root = insertAUX(self.root, key, self.cmp)
    
    def delete(self, key):
        self.root = deleteAUX(self.root, key, self.cmp)

    def getBelowAbove(self, key):
        findPreSucAUX.pre = None
        findPreSucAUX.suc = None

        findPreSucAUX(self.root, key, self.cmp)

        below = findPreSucAUX.pre
        above = findPreSucAUX.suc

        return below, above
        
# A utility function to do inorder traversal of BST
def inOrderAUX(root):
    if root is not None:
        inOrderAUX(root.left)
        print(root.key, end=' ')
        inOrderAUX(root.right)
 
# A utility function to insert a new node with given key in BST
def insertAUX(node, key, cmp):
    # If the tree is empty, return a new node
    if node is None:
        return Node(key)
 
    # Otherwise, recur down the tree
    if cmp(key, node.key) < 0:
        node.left = insertAUX(node.left, key, cmp)
    else:
        node.right = insertAUX(node.right, key, cmp)
 
    # return the (unchanged) node pointer
    return node
 
# Given a binary search tree and a key, this function
# deletes the key and returns the new root
def deleteAUX(root, key, cmp):
    # Base case
    if root is None:
        return root
 
    # Recursive calls for ancestors of
    # node to be deleted
    if cmp(root.key, key) > 0:
        root.left = deleteAUX(root.left, key, cmp)
        return root
    elif cmp(root.key, key) < 0:
        root.right = deleteAUX(root.right, key, cmp)
        return root
 
    # We reach here when root is the node
    # to be deleted.
 
    # If one of the children is empty
    if root.left is None:
        temp = root.right
        del root
        return temp
    elif root.right is None:
        temp = root.left
        del root
        return temp
 
    # If both children exist
    else:
        succParent = root
 
        # Find successor
        succ = root.right
        while succ.left is not None:
            succParent = succ
            succ = succ.left
 
        # Delete successor.  Since successor
        # is always left child of its parent
        # we can safely make successor's right
        # right child as left of its parent.
        # If there is no succ, then assign
        # succ.right to succParent.right
        if succParent != root:
            succParent.left = succ.right
        else:
            succParent.right = succ.right
 
        # Copy Successor Data to root
        root.key = succ.key
 
        # Delete Successor and return root
        del succ
        return root

def findPreSucAUX(root, key, cmp):
    # Base Case
    if root is None:
        return
 
    # If key is present at root
    if cmp(root.key, key) == 0:
 
        # the maximum value in left subtree is predecessor
        if root.left is not None:
            tmp = root.left 
            while(tmp.right):
                tmp = tmp.right 
            findPreSucAUX.pre = tmp 
 
        # the minimum value in right subtree is successor
        if root.right is not None:
            tmp = root.right
            while(tmp.left):
                tmp = tmp.left 
            findPreSucAUX.suc = tmp 
 
        return
 
    # If key is smaller than root's key, go to left subtree
    if cmp(root.key, key) > 0:
        findPreSucAUX.suc = root 
        findPreSucAUX(root.left, key, cmp)
 
    else: # go to right subtree
        findPreSucAUX.pre = root
        findPreSucAUX(root.right, key, cmp)