import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import time
import math
import itertools
from tqdm import tqdm

from ExtractingActivations import VerboseExecution
from CorrelatingActivations import getLayerCorrelation

#Layer 38 is the output layer
#Layers 32, 35, and 38 are the fully connected layers.
#The fully connected layers do not perform their activation functions.
#The activation functions are all ReLU and are the layers 33 and 36.
#Layers 34 and 37 are dropout layers. The dropout layers do nothing outside of training.



#flatten = lambda t: [item for sublist in t for item in sublist]

def isSubsetOfAnySet(setToTest, groupOfSets):

    for set in groupOfSets:
        if setToTest.issubset(set) and setToTest != set:
            return True
    return False

def doAllHaveHighCorrelation(array, correlations):
    for a, b in itertools.combinations(array, 2):
        if not correlations[a, b]:
            return False
    return True

def makeGroupings(layerCorrelations, threshold):#outputs a list of lists that contain the indicies of the neurons to be grouped
    numberOfNeurons = layerCorrelations.shape[0]

    shouldMerge = np.logical_or((layerCorrelations > threshold), np.identity(numberOfNeurons))#determines which neurons should merge (this is done here once rather than on site multiple times)
    fun = lambda shouldMergeArray: np.sum(shouldMergeArray)
    shouldMergeSums = np.apply_along_axis(fun, 0, shouldMerge)#the number of other neurons each neuron should merge with
    shouldMerge = torch.tensor(shouldMerge)
    shouldMergeSums = torch.tensor(shouldMergeSums)

    #makes all the groups. Including groups that are a subset of larger groups
    groups = []
    for i in range(2, numberOfNeurons + 1):
        validNeurons = ((shouldMergeSums >= i) * torch.tensor(np.indices((numberOfNeurons,))[0] + 1))#filters neurons to be combined based on the number of neurons each neuron is supposed to merge with. (if a neuron only merges with two other, don't check for the set of three it would be a part of. Because such a group doesn't exist)
        validNeurons = validNeurons[validNeurons != 0] - 1
        if(validNeurons.size(0) == 0):#If the possiblity of a group of size "i" doesn't exist stop checking for groups
            break
        print("group size: ", i, "valid neurons: ", validNeurons.size(0))
        pbar = tqdm(total=math.comb(validNeurons.size(0), i))

        for group in itertools.combinations(validNeurons.tolist(), i):#goes through combinations
            if doAllHaveHighCorrelation(group, shouldMerge):
                groups.append(group)
            pbar.update(1)
        pbar.close()
    
    #finds groups of one that are not a subset of a larger group
    #these two lines find the indicies of neurons that are only valid to merge with themselves.
    leftOverNeurons = ((shouldMergeSums == 1) * torch.tensor(np.indices((numberOfNeurons,))[0] + 1))
    leftOverNeurons = leftOverNeurons[leftOverNeurons != 0] - 1

    groups += [(neuron.tolist(),) for neuron in leftOverNeurons]#adds the groups of one to the list of groups

    #to make changed
    #filter out any groups that are a subset of any other group
    groups = [frozenset(group) for group in groups]
    groupsFilter = [not isSubsetOfAnySet(group, groups) for group in groups]
    tempGroups = []
    for i in range(len(groups)):
        if groupsFilter[i]:
            tempGroups.append(tuple(groups[i]))
    #groups = [tuple(group) for group in groups]
    groups = tempGroups
    del tempGroups, groupsFilter
    return groups




#EXAMPLE

#dummy_input = torch.rand(100, 3, 224, 224)
#VGG16 = VerboseExecution(models.vgg16())

#VGG16(dummy_input)

#linear1Correlations = getLayerCorrelation(VGG16, 32)
#linear2Correlations = getLayerCorrelation(VGG16, 35)
#linear3Correlations = getLayerCorrelation(VGG16, 38)




#start = time.perf_counter()
#makeGroupings(linear1Correlations, 0.9)
#print("completed 1")
#makeGroupings(linear2Correlations, 0.9)
#print("completed 2")
#makeGroupings(linear3Correlations, 0.9)
#print("completed 3")
#end = time.perf_counter()
#print(end - start)
