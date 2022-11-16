import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
import os
import sympy as sp


def arrayOfFolderImages(filepath):
    arr=[]
    directoryList=os.listdir(filepath)
    for dr in directoryList:
        str=filepath+"\\"+dr
        img=cv.imread(filepath+"\\"+dr)
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img=cv.resize(img,(240,240))
        arr.append(img)
    return arr

img1=cv.imread("D:/Kuliah/AlGeo/Tubes/Algeo02-21047/src/test/face.jpg")
img2=cv.imread("D:/Kuliah/AlGeo/Tubes/Algeo02-21047/src/test/Adriana Lima3_152.jpg")
img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
img1=cv.resize(img1,(240,240))
img2=cv.resize(img2,(240,240))
print(img1.shape)

def createImgVector(image):
    h,w=image.shape
    vector=np.zeros((1,h*w))
    count=0
    for i in image:
        for j in i:
            vector[count]+=j
    return vector


def meanOfVectors(imgArr):
    sum=np.zeros((len(imgArr),1))
    for i in range(len(imgArr)):
        for j in range(len(imgArr[i])):
            sum[i][0]+=imgArr[i][j]
    print(sum.shape)
    sum/=len(imgArr)
    return sum

def meanOfVectorsRGB(imgArr):
    sum=np.zeros((len(imgArr[0]),3))
    for i in range(len(imgArr)):
        for j in range(len(imgArr[i])):
            for k in range(3):
                sum[j][k]+=imgArr[i][j][k]
    sum/=len(imgArr)
    return sum

#  vector gambar 1: [[1,2,3],[1,2,3],[1,2,3]......NxM]

def subtractAllImages(imgArr,mean):
    for i in range(len(imgArr)):
        for j in range(len(imgArr[i])):
            imgArr[i][j]-=mean[j]
    return imgArr

def subtractAllImagesRGB(imgArr,mean):
    for i in range(len(imgArr)):
        for j in range(len(imgArr[i])):
            for k in range(3):
                imgArr[i][j][k]-=mean[j][k]
    return imgArr  

def getCovarice(vecArr):
    print(vecArr.T.shape)
    print(vecArr.shape)
    covarice=np.matmul(vecArr.T,vecArr)
    return covarice

def eigen(covarice):
    h,w=covarice.shape
    id=np.identity(h)
    eigenId=id*x
    sum=eigenId-covarice
    sumsp=sp.Matrix(sum)
    determinant=sumsp.det()
    temp,factor=sp.factor_list(determinant)
    print(len(factor))
    # for fac,amount in factor:
    #     eigen=sp.solveset(sp.Eq(fac,0),x)

imgArr=arrayOfFolderImages("D:/Kuliah/AlGeo/Tubes/Algeo02-21047/src/dataset/pins_Adriana Lima")
vecArr=np.empty((len(imgArr[0])*len(imgArr[0]),len(imgArr)))
count=1
temp=createImgVector(imgArr[0])
temp=np.asarray(temp)
temp=temp.T
i=1
while(i<len(imgArr)):
    imgtemp=createImgVector(imgArr[i])
    imgtemp=np.asarray(imgtemp)
    imgtemp=imgtemp.T
    print("IMGTEMP SIZE")
    print("Processing image",count)
    vecArr=np.concatenate((temp,imgtemp),axis=1)
    temp=vecArr
    count+=1
    i+=1

print(vecArr.shape)
print("DONE")
meanVec = meanOfVectors(vecArr)
print("meanvec")
print(meanVec.shape)
newVec=subtractAllImages(vecArr,meanVec)
print("newvec")
print(meanVec.shape)
covarice=getCovarice(newVec)
print("Covarice has been calculated")
print(covarice.shape)
shape=int(np.sqrt(len(newVec[0])))
testImg=np.zeros((shape,shape,3))
count=0
for i in range(shape):
    for j in range(shape):
        testImg[i][j]=meanVec[count]
        count+=1
# print(testImg.shape)
cv.imwrite("D:/Kuliah/AlGeo/Tubes/Algeo02-21047/src/test/hasil.jpg",testImg)
