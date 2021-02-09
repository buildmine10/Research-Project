from os import listdir
from os.path import isfile, join
from torch.utils import data
from torchvision import transforms, datasets
import torch
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision.transforms.transforms import ConvertImageDtype
from tqdm import tqdm


transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224)                #[3]
])

def makeDirectory(getLocation, putLocation, start):
    print("Loading")
    files = [f for f in tqdm(listdir(getLocation)[start:]) if isfile(join(getLocation, f))]
    print("Saving")
    for file in tqdm(files):
        transform(Image.open("D:/Python programs/Research Project/Images/" + file)).save(putLocation + "/" + file)


makeDirectory("D:/Python programs/Research Project/Images", "D:/Python programs/Research Project/Resized Images", 0)