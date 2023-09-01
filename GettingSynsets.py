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

    synsets = soup.body.find_all(href=isImagenet)
    names = [element.contents for element in synsets]
    synsets = [element["href"] for element in synsets]
    synsets = [element[element.find("=") + 1:] for element in synsets]
    
    return [synsets, names]

def urlToImage(url):
    resp = urllib.request.urlopen(url, timeout=60)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

#takes a list of image urls and compiles them into a folder
def makeDirectory(urls, path, start):
    progress = start
    for url in tqdm(urls[start:]):
        progress += 1
        try:
            I = urlToImage(url[0])
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = path + "\img" + str(progress) + "_" + url[1] + '.jpg'#create a name of each image
                cv2.imwrite(save_path,I)
        except:
            None

#gets the urls of the synset 
def loadSynset(synset):
    try:
        page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synset)
        soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line
    except:
        soup = ""
    
    string = str(soup)
    urls = string.split('\r\n')
    return urls


synsetInfo = get2014Synsets()
classes = np.loadtxt("D:\Python programs\Research Project\imagenet_classes.txt", dtype="str")

print("Getting synset id, label pairs.")
#for some reason not all synsets are found to have a matching name
synsets = []#synset id, class label
for index in tqdm(range(len(synsetInfo[0]))):
    name = synsetInfo[1][index][0][:synsetInfo[1][index][0].find(",")]
    synsetID = synsetInfo[0][index]
    for _class in classes:
        if(_class[1].replace("_", " ").lower() == name.replace("_", " ").lower()):
            synsets.append((synsetID, int(_class[0][:-1])))


print("Getting synset urls.")
urls = []
for synset in tqdm(synsets):
    temp_urls = loadSynset(synset[0])
    temp = np.array([temp_urls, [synset[1]] * len(temp_urls)]).T
    urls.extend(temp.tolist())
    1100

print(len(urls))

#random.shuffle(urls)

print("Creating directory.")
makeDirectory(urls, "D:\Python programs\Research Project\Images", 477029)
