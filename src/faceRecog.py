import math
import numpy
from utility import *
from tesss import *
import cv2
import PIL
from PIL import Image
import scipy
from scipy import interpolate


def euclideanDistance (eigenfaces1,eigenfaces2):
    sumDiff = 0

    # Menghitung jumlah selisih kuadrat eigenface training image dengan eigenface dataset
    # for i in range (len(eigenfaces2)):
    #     sumDiff += (eigenfaces1[i] - eigenfaces2[i]) ** 2
    
    sumDiff = np.subtract(eigenfaces2,eigenfaces1)
    sumDiff = list(map(lambda x: pow(x,2), sumDiff))
    sumDiff = np.sum(sumDiff)

    # Menghitung jarak terpendek
    ek = math.sqrt(sumDiff)
    return ek

def facerecog (eigenfaces1,imgList): # timg = training image, imgList = image pada dataset
    shortestEuclidean = euclideanDistance(eigenfaces1,imgList[0])
    indexSim = 0
    similiarImage = imgList[0]
    # toleranceLevel = 3

    # Mencari euclidean terkecil antara training image dengan dataset
    for i in range (1,len(imgList)): 
        ek = euclideanDistance(eigenfaces1,imgList[i])
        print(ek)
        if (shortestEuclidean > ek):
            shortestEuclidean = ek
            similiarImage = imgList[i]
            indexSim = i

    return similiarImage,indexSim
img = cv2.imread("C:\\Users\\Muhammad Libri\\OneDrive - Institut Teknologi Bandung\\Documents\\Algeo02-21047\\src\\dataset\\pins_Adriana Lima\\Adriana Lima139_40.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(256,256))
if img is not None:
    
    img=np.reshape(img,-1)
a=readAllImgInFolder("C:\\Users\\Muhammad Libri\\OneDrive - Institut Teknologi Bandung\\Documents\\Algeo02-21047\\src\\dataset\\all_12")
b=getMean(a)
c=getDifference(a,b)
d=getCovariance(c)
val,vec=eigen(d,100)
val1,vec1=np.linalg.eig(d)
x=eigenFace(vec,c)
x2=eigenFace(vec1,c)
m=interpolate.interp1d([min(img),max(img)],[0,255])
p,i=facerecog(img,x)
pp,ii=facerecog(img,x2)
print(i)
img2=Image.fromarray(a[i].reshape(256,256))
img2.save("HASIL.png",format="PNG")
img3=Image.fromarray(a[ii].reshape(256,256))
img3.save("HASIL1.png",format="PNG")