import math
import numpy
from utility import *
from tesss import *
import cv2


def euclideanDistance (eigenfaces1,eigenfaces2):
    sumDiff = 0

    # Menghitung jumlah selisih kuadrat eigenface training image dengan eigenface dataset
    # for i in range (length):
    # sumDiff += (eigenfaces1[i] - eigenfaces2[i]) ** 2
    
    sumDiff = np.subtract(eigenfaces1,eigenfaces2)
    sumDiff = list(map(lambda x: pow(x,2), sumDiff))
    sumDiff = np.sum(sumDiff)

    # Menghitung jarak terpendek
    ek = math.sqrt(sumDiff)
    return ek

def facerecog (eigenfaces1,imgList): # timg = training image, imgList = image pada dataset
    shortestEuclidean = 0
    similiarImage = imgList[0]
    # toleranceLevel = 3

    # Mencari euclidean terkecil antara training image dengan dataset
    for image in imgList: 
        ek = euclideanDistance(eigenfaces1,image)
        if (shortestEuclidean > ek):
            shortestEuclidean = ek
            similiarImage = image

    return similiarImage
img = cv2.imread("C:\\Users\\Muhammad Libri\\OneDrive - Institut Teknologi Bandung\\Documents\\Algeo02-21047\\src\\dataset\\pins_Adriana Lima\\Adriana Lima17_70.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(256,256))
if img is not None:
    
    img=np.reshape(img,-1)
a=readAllImgInFolder("dataset\\all_12")
b=getMean(a)
c=getDifference(a,b)
d=getCovariance(c)
val,vec=eigen(d,1)
x=eigenFace(vec,c)
p=facerecog(img,x)
print(p)