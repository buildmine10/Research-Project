import torch
import torch.nn as nn
import torchvision.models as models

from ExtractingActivations import VerboseExecution
from CorrelatingActivations import getLayerCorrelation
from GroupByCorrelation import makeGroupings
from ReorganizeByGroupings import getMergedVGG16Weights
from ManipulatingWeights import VGGWithDifferentClassifier
import ValidatingModel
from tqdm import tqdm
import numpy as np

#Check if changes have been made.
#Check the difference in accuracy.
#Check the change in correlations.


#This is a temporary function
#Its content should just be free standing code
def makeModifiedModel(threshold):
    
    dummy_input = torch.rand(255, 3, 224, 224)
    #print(len(files[::1160]))
    #pic_input = ValidatingModel.loadImages("D:/Python programs/Research Project/Resized Images", files[::1160])#255 images

    VGG16 = VerboseExecution(models.vgg16(pretrained=True).eval())

    print("Running Input")
    VGG16(dummy_input)
    #VGG16(pic_input)


    print("Getting correlations")
    
    linear1Correlations = getLayerCorrelation(VGG16, 0)
    linear2Correlations = getLayerCorrelation(VGG16, 1)
    
    #linear1Correlations = []
    #linear2Correlations = []
    #
    #print("Getting correlations")
    ##Identification of neurons to be merged
    #for index in tqdm(range(0, len(files), 30)):
    #    input = ValidatingModel.loadImages("D:/Python programs/Research Project/Resized Images", files[index: index + 30]).cuda()
    #    VGG16(input)
    #    linear1Correlations.append(getLayerCorrelation(VGG16, 32))
    #    linear2Correlations.append(getLayerCorrelation(VGG16, 35))
    ##The third linear layer is does not get merged because it is the output layer.
    #linear1Correlations = np.concatenate(linear1Correlations)
    #linear2Correlations = np.concatenate(linear2Correlations)

    print("Making groupings")
    #Creation of merge groups
    linear1Groupings = makeGroupings(linear1Correlations, threshold)
    #linear1Groupings = [(0, 1), (1, 2, 3)]
    print("Finished grouping linear 1.")
    linear2Groupings = makeGroupings(linear2Correlations, threshold)
    #linear2Groupings = [(0, 1), (2,)]
    print("Finished grouping linear 2.")

    print("Making new layers")
    newParamters = getMergedVGG16Weights(linear1Groupings, linear2Groupings)

    #Construction of new classifier
    myLayers = []
    myLayers.append(nn.Linear(newParamters[0][0].size(1), newParamters[0][0].size(0)))
    myLayers.append(nn.ReLU(True))
    myLayers.append(nn.Linear(newParamters[1][0].size(1), newParamters[1][0].size(0)))
    myLayers.append(nn.ReLU(True))
    myLayers.append(nn.Linear(newParamters[2][0].size(1), newParamters[2][0].size(0)))

    myLayers[0].weight = nn.parameter.Parameter(newParamters[0][0])
    myLayers[2].weight = nn.parameter.Parameter(newParamters[1][0])
    myLayers[4].weight = nn.parameter.Parameter(newParamters[2][0])

    myLayers[0].bias = nn.parameter.Parameter(newParamters[0][1])
    myLayers[2].bias = nn.parameter.Parameter(newParamters[1][1])
    myLayers[4].bias = nn.parameter.Parameter(newParamters[2][1])

    #Construction of new network
    classifier = nn.Sequential(*myLayers)
    myModel = VGGWithDifferentClassifier(classifier)#VerboseExecution(VGGWithDifferentClassifier(classifier))
    #print(myModel)
    return myModel

def testModel(model):

    #dummy_input = torch.rand(253, 3, 224, 224)
    #print(len(files[::1160]))
    pic_input = ValidatingModel.loadImages("D:/Python programs/Research Project/Resized Images", files[::1160])#255 images

    model = VerboseExecution(models.vgg16(pretrained=True).eval())

    print("Running Input")
    #model(dummy_input)
    model(pic_input)


    print("Getting correlations")
    
    linear1Correlations = getLayerCorrelation(model, 0)
    linear2Correlations = getLayerCorrelation(model, 1)

    #linear1AverageCorrelation = np.sum(np.abs(linear1Correlations)) / (linear1Correlations.shape[0] * linear1Correlations.shape[1])
    #linear2AverageCorrelation = np.sum(np.abs(linear2Correlations)) / (linear2Correlations.shape[0] * linear2Correlations.shape[1])
    linear1AverageCorrelation = np.sum(linear1Correlations * linear1Correlations > 0) / (linear1Correlations.shape[0] * linear1Correlations.shape[1])
    linear2AverageCorrelation = np.sum(linear2Correlations * linear2Correlations > 0) / (linear2Correlations.shape[0] * linear2Correlations.shape[1])

    print(linear1AverageCorrelation)
    print(linear2AverageCorrelation)


f = open("output.txt", "a")
f.write("start")
f.write("\n")
f.write("\n")
f.write("\n")
f.write("\n")
f.close()

print("Loading")
folderPath = "D:/Python programs/Research Project/Resized Images"
files = ValidatingModel.getFiles(folderPath, 0, 646181)

#myModel = makeModifiedModel(0.9)
#torch.save(myModel, "D:/Python programs/Research Project/ModifiedVGG16UsingRandom0.9.pt")

#myModel.cuda()
#f = open("output.txt", "a")
#f.write(str(myModel))


#vgg16 = models.vgg16(pretrained=True).eval()
#vgg16.cuda()


#myModel = torch.load("D:/Python programs/Research Project/ModifiedVGG16UsingRandom0.9V2.pt")
#myModel.eval()

#testModel(myModel)


#print("Evaluating")
#print(len(files[::1150]))
#dummy_input = torch.rand(250, 3, 224, 224)
#pic_input = ValidatingModel.loadImages("D:/Python programs/Research Project/Resized Images", files[::1150])#308 images
#myModel(pic_input)
#print("done")
#print("Modified", ValidatingModel.evaluateModel(myModel, folderPath, files))

#print("Evaluating")
#print("Original", ValidatingModel.evaluateModel(vgg16, folderPath, files))


f = open("output.txt", "a")
print("starting Random 0.9")
#myModel = makeModifiedModel(0.8)
#torch.save(myModel, "D:/Python programs/Research Project/ModifiedVGG16UsingRandom0.8V2.pt")
myModel = torch.load("D:/Python programs/Research Project/ModifiedVGG16UsingRandom0.9V2.pt")
myModel.eval()
myModel.cuda()
f.write(str(myModel))
f.write("\n")
print("Evaluating")
evaluation = str(ValidatingModel.evaluateModel(myModel, folderPath, files))
f.write("Random 0.9   ")
f.write(evaluation)
f.write("\n")
f.write("\n")
f.write("\n")
f.write("\n")
print("Random 0.9", evaluation)
f.close()
testModel(myModel)
