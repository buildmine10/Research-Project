from torch.utils import data
from torchvision import transforms, datasets
import torch
from PIL import Image
import numpy as np
import torchvision.models as models
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import random

#transform = transforms.Compose([            #[1]
#    transforms.Resize(256),                    #[2]
#    transforms.CenterCrop(224),                #[3]
#    transforms.ToTensor(),                     #[4]
#    transforms.Normalize(                      #[5]
#    mean=[0.485, 0.456, 0.406],                #[6]
#    std=[0.229, 0.224, 0.225]                  #[7]
#)])

transform = transforms.Compose([
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
)])

def getFiles(path, start, end):
    files = [f for f in tqdm(listdir(path)[start: end]) if isfile(join(path, f))]
    random.shuffle(files)
    return files

def loadImages(path, files):
    imgs = []
    #for file in tqdm(files):
    for file in files:
        imgs.append(transform(Image.open(path + "/" + file)))
    #random.shuffle(imgs)
    return torch.stack(imgs)


def decodeResult(result):
    with open("D:/Python programs/Research Project/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(result, 1)
    percentage = torch.nn.functional.softmax(result, dim=1)[0] * 100
    return [classes[i] for i in index.tolist()]

def evaluateResults(results, files):
    files = np.array([f.split("_")[1].split(".")[0] for f in files])
    results = np.array([result.split(",")[0] for result in results])
    isSame = files == results
    return np.sum(isSame) / isSame.shape[0]

def guageAccuracy(model, path, files):
    imgs = loadImages(path, files).cuda()
    out = model(imgs)
    #out.cpu()
    #model.cpu()
    #imgs.cpu()
    accuracy = evaluateResults(decodeResult(out), files)
    return accuracy

def evaluateModel(model, path, files):
    accuracies = []
    for index in tqdm(range(0, len(files), 30)):
        accuracies.append(guageAccuracy(model, path, files[index : index + 30]))
    accuracies = np.array(accuracies)
    return np.sum(accuracies) / accuracies.shape[0]

#vgg16 = models.vgg16(pretrained=True).eval()
#vgg16.cuda()
#
#print("Loading")
#folderPath = "D:/Python programs/Research Project/Resized Images"
#files = getFiles(folderPath, 0, 100000)
#
#print("Evaluating")
#print(evaluateModel(vgg16, folderPath, files))
