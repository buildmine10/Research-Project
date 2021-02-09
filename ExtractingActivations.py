import torch
import torch.nn as nn
import torchvision.models as models

#Use VerboseExecution to get activations from all named layers
#Named layers are the important ones
#Unnamed layers are probably layers that only activate during training

#To get activations run the network and then access the variable "activations" from the VerboseExecution object
#"activations" is a python list





class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activations = []#stores the activations after being ran

        # Register a hook for each layer
        #for name, layer in self.model.named_children():
        #    layer.__name__ = name
        #    layer.register_forward_hook(
        #        lambda layer, _, output: print(f"{layer.__name__}: {output.size()}")
        #    )


        #I need to register a hook for every named layer in the model
        #If a layer has named children I need to go in those children and add hooks
        def addHook(module: nn.Module):
            for name, layer in module.named_children():
                if(len(list(layer.named_children())) == 0):
                    layer.__name__ = name #The layer's name should already be (name), right? Why does this need to be done?
                    layer.register_forward_hook(
                        lambda layer, _, output: self.activations.append(output)
                    )
                addHook(layer)
        addHook(self.model)


    def forward(self, x):
        self.activations = []
        return self.model(x)

    def getLayers(self):
        layers = []

        def findLayers(module: nn.Module):
            for name, layer in module.named_children():
                if(len(list(layer.named_children())) == 0):
                    layer.__name__ = name
                    layers.append(layer)
                findLayers(layer)

        findLayers(self.model)
        return layers

#EXAMPLE

#verbose_vgg = VerboseExecution(models.vgg16())
#dummy_input = torch.ones(10, 3, 224, 224)

#_ = verbose_vgg(dummy_input)
#for layer in verbose_vgg.activations:
#    print(layer.size())