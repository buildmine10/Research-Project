import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import scipy

from ExtractingActivations import VerboseExecution


def getLayerCorrelation(model: VerboseExecution, layerIndex):
    layerActivations = model.activations[layerIndex].cpu().detach().numpy()
    return np.nan_to_num(np.corrcoef(layerActivations, rowvar=False))

#EXAMPLE

#fully connected layers start at layer 32
#verbose_vgg = VerboseExecution(models.vgg16()).eval()
#dummy_input = torch.rand(100, 3, 224, 224)
#
#_ = verbose_vgg(dummy_input)
#
#
#layersThatMatter = [32, 35, 38]#These layers are the fully connected layers. The other layers do not have weights
#
#layers = verbose_vgg.getLayers()
#
#correlations1 = getLayerCorrelation(verbose_vgg, 32)
#correlations2 = getLayerCorrelation(verbose_vgg, 35)
#correlations3 = getLayerCorrelation(verbose_vgg, 38)
#
#print(correlations1.shape)
#print(correlations2.shape)
#print(correlations3.shape)