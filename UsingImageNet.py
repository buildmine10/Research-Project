from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib
from tqdm import tqdm
import random
import math
import re

def get2014Synsets():
    page = requests.get("http://image-net.org/challenges/LSVRC/2014/browse-synsets")
    soup = BeautifulSoup(page.content, 'html.parser')

    def isImagenet(href):
        return href and re.compile("imagenet.stanford.edu/synset").search(href)

    synsets = [element["href"] for element in soup.body.find_all(href=isImagenet)]
    synsets = [element[element.find("=") + 1:] for element in synsets]
    
    return synsets

def urlToImage(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def makeDirectory(urls, path):
    progress = 0
    for url in tqdm(urls):
        progress += 1
        try:
            I = urlToImage(url)
            if (len(I.shape))==3: #check if the image has width, length and channels
              save_path = path + "\img" + str(progress) + '.jpg'#create a name of each image
              cv2.imwrite(save_path,I)
        except:
            None
        
def loadSynset(synset):
    try:
        page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synset)
        soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line
    except:
        soup = ""
    
    string = str(soup)
    urls = string.split('\r\n')
    return urls

urls = []
label = 0
for synset in get2014Synsets()[0:10]:
    synsets = loadSynset(synset)
    temp = np.array([synsets, [label] * len(synsets)]).T
    urls.extend(temp.tolist())
    label += 1

print(len(urls))
print(urls[840])
#random.shuffle(urls)

#urls.extend(loadSynset("n04194289"))
#urls.extend(loadSynset("n02834778"))
#random.shuffle(urls)
#print(len(urls))
#
#trainingSetSize = math.floor(len(urls) * 0.9)
#
#makeDirectory(urls[:trainingSetSize], "D:\Python programs\Research Project\TrainingSet")
#makeDirectory(urls[trainingSetSize:], "D:\Python programs\Research Project\ValidationSet")
makeDirectory(urls[0], "D:\Python programs\Research Project\TrainingSet")

