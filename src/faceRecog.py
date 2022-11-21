import math
import numpy
from utility import *
from tesss import *
import cv2
import PIL
from PIL import Image
import scipy
from scipy import interpolate


def euclideanDistance (weight1,weight2):
    sumDiff = 0

    # Menghitung jumlah selisih kuadrat eigenface training image dengan eigenface dataset
    # for i in range (len(eigenfaces2)):
    #     sumDiff += (eigenfaces1[i] - eigenfaces2[i]) ** 2
    
    sumDiff = np.subtract(weight1,weight2)
    sumDiff = list(map(lambda x: pow(x,2), sumDiff))
    sumDiff = np.sum(sumDiff)
    # sumDiff = (weight1-weight2) ** 2

    # Menghitung jarak terpendek
    ek = np.sqrt(sumDiff)
    return ek

def facerecog (img,eigenfaces,normimgList,mean): # timg = training image, imgList = image pada dataset
    # shortestEuclidean = euclideanDistance(img,eigenfaces[0])
    norm1 = normalized(img,mean)
    teigenfaces = np.transpose(eigenfaces)
    weigth1 = np.dot(norm1,teigenfaces)
    weight2 = np.dot(normimgList,teigenfaces)
    shortestEuclidean = euclideanDistance(weigth1[0],weight2[0])
    indexSim = 0
    #similiarImage = [0]
    # toleranceLevel = 3

    # Mencari euclidean terkecil antara training image dengan dataset
    for i in range (1,len(eigenfaces)):
        ek = euclideanDistance(weigth1[0],weight2[i])
        if (shortestEuclidean > ek):
            shortestEuclidean = ek
            indexSim = i

    print(shortestEuclidean)
    return indexSim

def normalized(img,mean):
    norm = np.subtract(img,mean)
    return norm

# a=readAllImgInFolder("C:\\Users\\Muhammad Libri\\Downloads\\Algeo02-21047\\src\\Data Set New")
# b=getMean(a)
# c=getDifference(a,b)
# d=getCovariance(c)
# val,vec=eigen(d,1)
# #val1,vec1=np.linalg.eig(d)
# x=eigenFace(vec,c)
# x2=eigenFace(vec1,c)
# m=interpolate.interp1d([min(img),max(img)],[0,255])
# pa=m(img)
# p,i=facerecog(pa,x)
# pp,ii=facerecog(pa,x2)
# print(i)
# img2=Image.fromarray(a[i].reshape(256,256))
# img2.save("HASIL.png",format="PNG")
# img3=Image.fromarray(a[ii].reshape(256,256))
# img3.save("HASIL1.png",format="PNG")
# img = cv2.imread("C:\\Users\\Muhammad Libri\\Downloads\\Algeo02-21047\\src\\IMG20201216191636.jpg",cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(256,256))
# if img is not None:
#     img=np.reshape(img,-1)

# imgS=Image.fromarray(img.reshape(256,256))
# imgS.save("Search.png",format="PNG")
# # image = []
# image.append(img)
#b1=getMean(image)

# c1=getDifference(image,mean)
# d1=getCovariance(c1)
# val1,vec1=eigen(d1,1)
# #val1,vec1=np.linalg.eig(d)
# x1=eigenFace(vec1,c1)
# print(image)
# sum = sum(img)
# mean = sum/len(img)
# c1=getDifference(image,mean)
# print(c1)
# d1=np.matmul(c1,np.transpose(c1))
# print(d1)
# val1,vec1=np.linalg.eig(d1)
# #x1=eigenFace(vec1,c1)
# # print(img.shape)
# # img=[img]
# # image=[]
# # image.append(img)
# # print(len(image),len(image[0]))
# # oo,op=img.shape
# # print(oo,op)
# # b1 = getMean(img)
# # c1 = getDifference(img,b1)
# # d1=getCovariance(image)
# # val1,vec1=eigen(d1,1)
# # x1=eigenFace(vec1,np.transpose(c))
# print(vec1)
# v = vec1[0]
# print(len(v))
# print(len(np.subtract(img,mean)))
# o = np.cross(v,np.subtract(img,mean))

# s = normalized(img)
# print(s)
#print(b)
# imgM = Image.fromarray(b.reshape(256,256))
# imgM.save("MEAN.png",format="PNG")
# print(len(x))
# image = [img]
# i = facerecog(image,x,c,b)
# print(i)
# img2=Image.fromarray(a[i].reshape(256,256))
# img2.save("HASIL.png",format="PNG")