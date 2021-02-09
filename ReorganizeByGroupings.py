import torch
import torch.nn as nn
import torchvision.models as models

from ExtractingActivations import VerboseExecution
from CorrelatingActivations import getLayerCorrelation
from GroupByCorrelation import makeGroupings



def getMergedVGG16Weights(linear1Groupings, linear2Groupings):
    tempVGG = VerboseExecution(models.vgg16(pretrained=True).eval())
    layers = tempVGG.getLayers()

    #Gets original values
    linear1Weights = layers[32].weight
    linear2Weights = layers[35].weight
    linear3Weights = layers[38].weight
    linear1Biases = layers[32].bias
    linear2Biases = layers[35].bias

    print("Linear1 Size: ", linear1Weights.size())
    print("Linear2 Size: ", linear2Weights.size())
    print("Linear3 Size: ", linear3Weights.size())

    linear2Weights, linear2Biases, linear3Weights = reorganizeLayer(linear2Weights, linear2Biases, linear3Weights, linear2Groupings)
    linear1Weights, linear1Biases, linear2Weights = reorganizeLayer(linear1Weights, linear1Biases, linear2Weights, linear1Groupings)

    print()
    print("Linear1 new size: ", linear1Weights.size())
    print("Linear2 new size: ", linear2Weights.size())
    print("Linear3 new size: ", linear3Weights.size())

    return ((linear1Weights, linear1Biases), (linear2Weights, linear2Biases), (linear3Weights, layers[38].bias))#the last one is the biases of the linear3 (they are unmodified)




def reorganizeLayer(layerWeights, layerBiases, nextLayerWeights, layerGroupings):
    #The layer groupings determine which weights get merged together
    #layer is the current layer
    #next layer is the next layer
    #For layer each group is a neuron, and the incoming weights of those neurons are the averages of the weights found in the group
    #The outgoing weights are stored in next layer
    #For next layer each group represents one neuron to recieve data from
    #The weights of next layer are the averages of the outgoing weights from the neurons in each group

    layerNeuronWeights = []#Each list element is a neuron. In each element the first dimension is are the individual neurons whose weights need to be merged, and the second dimension contains the weights of each neuron.
    nextLayerIncomingWeights = []#Each list element represents the incoming weights form one neuron. The second dimension of each element contains the weights that need to be merged in order to find the weights of the incoming neuron.
    layerBiasGroups = []#Each list element is the bias for a single neuron
    for grouping in layerGroupings:
        layerNeuronWeights.append(layerWeights[grouping, :])#gathers all incoming weights in the group
        nextLayerIncomingWeights.append(nextLayerWeights[:, grouping])#gathers all outgoing weights in the group
        layerBiasGroups.append(layerBiases[grouping, ...])

    layerWeights = []
    for neuron in layerNeuronWeights:
        layerWeights.append(torch.sum(neuron, 0) / neuron.size(0))#averages the weights of all neurons in each group for each group

    layerWeights = torch.tensor([neuron.cpu().detach().tolist() for neuron in layerWeights])#converts to a tensor of size (number of neurons, number of neurons in previous layer)

    nextLayerWeights = []
    for neuron in nextLayerIncomingWeights:
        nextLayerWeights.append(torch.sum(neuron, 1) / neuron.size(1))#averages the weights of all neurons in each group for each group

    nextLayerWeights = torch.tensor([neuron.cpu().detach().tolist() for neuron in nextLayerWeights]).permute(1, 0)#converts to a tensor of size (number of neurons, number of neurons in previous layer)
    #It also swaps dimensions 0 and 1 because Pytorch needs the dimensions to be in the format (to, from).
    #After swapping the dimensions represent (to, from). Prior to swapping they represent (from, to).

    layerBiases = []
    for neuron in layerBiasGroups:
        layerBiases.append(torch.sum(neuron, 0) / neuron.size(0))#averages the biases together

    layerBiases = torch.tensor([neuron.cpu().detach().tolist() for neuron in layerBiases])#converts to a tensor of size (number of neurons)

    return (layerWeights, layerBiases, nextLayerWeights)





#EXAMPLE

#dummy_input = torch.rand(50, 3, 224, 224)
#VGG16 = VerboseExecution(models.vgg16().eval())
#
#VGG16(dummy_input)
#
#c = getLayerCorrelation(VGG16, 32)
#print(c.shape)
#
##The third linear layer is does not get merging because it is the output layer.
#
#linear1Correlations = getLayerCorrelation(VGG16, 32)
#linear2Correlations = getLayerCorrelation(VGG16, 35)
#
#linear1Groupings = makeGroupings(linear1Correlations, 0.9)
##linear1Groupings = [(0, 1), (1, 2, 3)]
#print("Finished grouping linear 1.")
#linear2Groupings = makeGroupings(linear2Correlations, 0.9)
##linear2Groupings = [(0, 1), (2,)]
#print("Finished grouping linear 2.")
#
#print("Linear1 Groupings sample: ", linear1Groupings[:10])
#print("Linear2 Groupings sample: ", linear2Groupings[:10])
#
#newVGG16ClassifierParameters = getMergedVGG16Weights(linear1Groupings, linear2Groupings)
#print([layer.size() for layer in newVGG16ClassifierParameters])