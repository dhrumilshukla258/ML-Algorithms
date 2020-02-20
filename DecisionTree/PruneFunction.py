import numpy as np
import math as math
from DecisionTree import DecisionTree

# Purning Algorithm by Classification Performance on Validation Set
# Count leaves in a tree
def CountLeaves(dt):
    if dt.isLeaf:
        return 1
    else:
        n = 0
        for val,child in dt.children.items():
            n += CountLeaves(child)
    return n

# Check if a node is a twig
def isTwig(dt):
    for val,child in dt.children.items():
        if not child.isLeaf:
            return False
    return True

# First create an empty list of error counts at nodes
def CreateNodeList( dt, nodeError={} ):
    nodeError[dt] = 0
    for val,child in dt.children.items():
        CreateNodeList(child,nodeError)
    return nodeError

# Pass a single instance down the tree and note node errors
def ClassifyValidationDataInstance( dt, validationDataInstance, nodeError ):
    labels = validationDataInstance[len(validationDataInstance)-1]
    if dt.majorityClass != labels:
        nodeError[dt] += 1
    if not dt.isLeaf:
        childNode = dt.children[ validationDataInstance[dt.bestAttribute] ]
        ClassifyValidationDataInstance( childNode, validationDataInstance, nodeError )
    return

# Count total node errors for validation data
def ClassifyValidationData(dt, validationData):
    nodeErrorCounts = CreateNodeList(dt)
    for instance in validationData:
        ClassifyValidationDataInstance(dt, instance, nodeErrorCounts)
    return nodeErrorCounts

# Second pass:  Create a heap with twigs using nodeErrorCounts
def CollectTwigsByErrorCount(dt, nodeErrorCounts, heap=[]):
    if isTwig(dt):
        # Count how much the error would increase if the twig were trimmed
        twigErrorIncrease = nodeErrorCounts[dt]
        for val,child in dt.children.items():
            twigErrorIncrease -= nodeErrorCounts[child]
        heap.append([twigErrorIncrease, dt])
    else:
        for val,child in dt.children.items():
            CollectTwigsByErrorCount(child, nodeErrorCounts, heap)
    return heap

# Third pass: Prune a tree to have nLeaves leaves
# Assuming heappop pops smallest value
def PruneByClassificationError(dt, validationData, nLeaves = -1):
    # First obtain error counts for validation data
    nodeErrorCounts = ClassifyValidationData(dt, validationData)
    
    # Get Twig Heap
    twigHeap = CollectTwigsByErrorCount(dt, nodeErrorCounts)

    totalLeaves = CountLeaves(dt)
    
    
    while totalLeaves > nLeaves:
        #Find index of minimum value of Twig from the Heap
        minVal = math.inf
        twigHeap_index = -1
        loop = 0
        for x in twigHeap:
            if x[0] < minVal:
                twigHeap_index = loop
            loop+=1
        
        if twigHeap[twigHeap_index][0] > 0 and nLeaves == -1:
            break
        
        twig = twigHeap[twigHeap_index][1]
        twigHeap.pop(twigHeap_index)
        totalLeaves -= (len(twig.children) - 1) 
        parent = twig.parent
        
        #Removing Twig from the original tree
        dt.RemoveChild( twig )
        
        #print ( twig )
        # Check if the parent is a twig and, if so, put it in the heap
        if isTwig(parent):
            twigErrorIncrease = nodeErrorCounts[parent]
            for val,child in parent.children.items():
                twigErrorIncrease -= nodeErrorCounts[child]
            twigHeap.append([twigErrorIncrease, parent])
    
    return dt
