import torch
import torch.nn as nn
import torchvision.models as models

from ExtractingActivations import VerboseExecution

#It replaces the end of the VGG16 network with whatever you give it.
class VGGWithDifferentClassifier(nn.Sequential):
    def __init__(self, classifier):
        super(VGGWithDifferentClassifier, self).__init__()
        vgg = models.vgg16(pretrained=True).eval()
        self.features = list(vgg.named_children())[0][1]
        self.avgpool = list(vgg.named_children())[1][1]
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1, -1)
        return self.classifier(x)


#EXAMPLE 
#Example may be broken VGGWithDifferentClassifier.vgg.model does not exist 
#It is now just VGGWithDifferentClassifier.vgg
#try replacing instances of vgg.model with vgg in the example code

#vgg = VerboseExecution(models.vgg16()).eval()
##print(vgg)
#layers = vgg.getLayers()[32:: 3]
###layers = vgg.getLayers()[32:]
##print(layers)
#
#layerSizes = [100, 50, 25]
#myLayers = []
#myLayers.append(nn.Linear(layers[0].weight.size()[1], layerSizes[0]))
#myLayers.append(nn.ReLU(True))
#myLayers.append(nn.Linear(layerSizes[0], layerSizes[1]))
#myLayers.append(nn.ReLU(True))
#myLayers.append(nn.Linear(layerSizes[1], layerSizes[2]))
#
#classifier = nn.Sequential(*myLayers)
###classifier = nn.Sequential(*layers)
#
#
#myModel = VerboseExecution(VGGWithDifferentClassifier(classifier))
##print(myModel)
#
#dummy_input = torch.rand(10, 3, 224, 224)
#myModel(dummy_input)
#
#a = myModel.activations
#print(a[32].size())
#print(a[34].size())
#print(a[36].size())