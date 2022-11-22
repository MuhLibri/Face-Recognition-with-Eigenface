import math
import numpy
from tesss import *
import cv2
import PIL
from PIL import Image
import scipy
from scipy import interpolate


def euclideanDistance (weight1,weight2):
    # Menghitung jumlah selisih kuadrat
    sumDiff = 0   
    sumDiff = np.subtract(weight1,weight2)
    sumDiff = list(map(lambda x: pow(x,2), sumDiff))
    sumDiff = np.sum(sumDiff)

    # ek = jarak euclidean
    ek = np.sqrt(sumDiff)
    return ek

def facerecog (img,eigenfaces,normimgList,mean): 
# img = gambar yang dicari, eigenfaces = matriks eigenface, normimgList = list gambar dari dataset yang telah dinormalisasi, mean = matriks rata-rata
    norm1 = normalized(img,mean)
    teigenfaces = np.transpose(eigenfaces)
    weigth1 = np.dot(norm1,teigenfaces)
    weight2 = np.dot(normimgList,teigenfaces)
    shortestEuclidean = euclideanDistance(weigth1[0],weight2[0])
    indexSim = 0
    maxEuclidean = shortestEuclidean

    # Mencari euclidean terkecil antara bobot image dengan bobot dataset
    for i in range (1,len(eigenfaces)):
        ek = euclideanDistance(weigth1[0],weight2[i])
        if (shortestEuclidean > ek):
            shortestEuclidean = ek
            indexSim = i
        if (maxEuclidean < ek):
            maxEuclidean = ek

    toleranceLevel = maxEuclidean/15

    # Mengembalikan -1 jika tidak ada foto yang cocok       
    if (shortestEuclidean > toleranceLevel):
        indexSim = -1

    return indexSim

def normalized(img,mean):
    norm = np.subtract(img,mean)
    return norm