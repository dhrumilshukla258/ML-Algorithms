import numpy as np
import math as math

# Segregating out instances that take a particular value
# attributearray is an N x 1 array.
def segregate(attributearray, value):
    outlist = []
    for i in range ( len(attributearray) ):
        if attributearray[i] == value:
            outlist.append(i)  #  Append "i" to outlist
    return outlist


def computeEntropy(labels):
    entropy = 0
    UniqueValuesInLabels =  np.unique( labels )
    for i in UniqueValuesInLabels:
        probability_i = len(segregate(labels, i)) / len(labels)
        entropy -= probability_i * math.log2(probability_i)
    return entropy

def mostFrequentlyOccurringValue(labels):
    bestCount = -math.inf
    bestId = None
    UniqueValuesInLabels =  np.unique( labels )
    for i in UniqueValuesInLabels:
       
        count_i = len(segregate(labels,i))
        if count_i > bestCount:
            bestCount = count_i
            bestId = i
    return bestId

def computeVarianceImpurity(labels):
    vi = 1
    UniqueValuesInLabels =  np.unique( labels )
    for i in UniqueValuesInLabels:
        probability_i = len(segregate(labels, i)) / len(labels)
        vi *= probability_i
    return vi

def InformationGainByVI( S, X ):
    vi = computeVarianceImpurity( S )
    UniqueValuesofX = np.unique(X)
    for Y in UniqueValuesofX:
        ids = segregate( X, Y )
        pr_x = len(ids) / len(S)
        vi_Sx = computeVarianceImpurity( S[ids] )
        vi -= pr_x * vi_Sx
    return vi

def InformationGainByEntropy( S, X ):
    vi = computeEntropy( S )
    UniqueValuesofX = np.unique(X)
    for Y in UniqueValuesofX:
        ids = segregate( X, Y )
        pr_x = len(ids) / len(X)
        vi_Sx = computeEntropy( S[ids] )
        vi -= pr_x * vi_Sx
    return vi