from zipfile import ZipFile
import cv2
import os

def extractZip(zipName,path):
    with ZipFile(zipName,'r') as zip:
        zip.printdir()
        zip.extractall(path)

def readAllImgInFolder(path):
    imglist=[]
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path,file))
        if img is not None:
            imglist.append(img)
    return imglist
