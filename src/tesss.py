import os
import time

import PIL.Image
from cv2 import  *
import cv2
from PIL import *
import  numpy as np
def readAllImgInFolder(path):
    imglist=[]
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(256,256))
        if img is not None:
            img=np.reshape(img,-1)
            imglist.append(img)
    # print(len(imglist[0]))
    return imglist
def getMean(imglist):
    l=len(imglist)
    sum=imglist[0]
    for i in range (1,l):
        sum=np.add(sum,imglist[i])
    sum=(sum/l)
    return sum
def getDifference(imglist,mean):
    diff=[]
    for img in imglist:
        diff.append(np.subtract(img,mean))
    return diff
def getCovariance(difference):
    # a=[]
    # for dif in difference:
    #     a.append(dif)
    at = np.transpose(difference)
    # print(a)
    # print(at)
    at=np.transpose(a)
    covariance=np.matmul(difference,at)
    covariance = (covariance/len(difference))
    # l=len(difference)
    # covariance=[]
    # for dif in difference:
    #     at=np.transpose(dif)
    #     print(dif)
    #     print(at)
    #     print(np.matmul(dif,at))
        # covariance.append(np.matmul(dif,at))
    # covariance = (covariance / len(difference))
    return covariance
def QR_factorisation_Householder_double(A):
    """Perform QR factorisation in double floating-point precision.
    :param A: The matrix to factorise.
    :type A: :py:class:`numpy.ndarray`
    :returns: The matrix Q and the matrix R.
    :rtype: tuple
    """
    A = np.array(A, 'float')

    n, m = A.shape
    V = np.zeros_like(A, 'float')
    for k in range(n):
        V[k:, k] = A[k:, k].copy()
        V[k, k] += np.sign(V[k, k]) * np.linalg.norm(V[k:, k], 2)
        V[k:, k] /= np.linalg.norm(V[k:, k], 2)
        A[k:, k:] -= 2 * np.outer(V[k:, k], np.dot(V[k:, k], A[k:, k:]))
    R = np.triu(A[:n, :n])

    Q = np.eye(m, n)
    for k in range((n - 1), -1, -1):
        Q[k:, k:] -= np.dot((2 * (np.outer(V[k:, k], V[k:, k]))), Q[k:, k:])
    return Q, R

def eigen(A,maxiter=1000):
    n=len(A)
    U = np.identity(n)
    eigenV=[]
    for i in range(maxiter):
        pel=[]
        Q,R = QR_factorisation_Householder_double(A)
        A=np.matmul(R,Q)
        U=np.matmul(U,Q)
    for vec in U:
        eigenV.append(vec)
    return np.diag(A),eigenV;
def eigenFace(eigenV,difference):
    eigenfaces=[]
    for i in range (len(difference)):
        eigenfaces.append(np.dot(eigenV[i],difference))
        # print(len(eigenfaces[i]))
    return eigenfaces
def saveEigenFaces(eigenfaces):
    print(len(eigenfaces[0]))
    for i in range(len(eigenfaces)):
        img=PIL.Image.fromarray(eigenfaces[i].reshape(256,256))
        img=img.convert("L")
        img.save("/hasil/"+str(i)+".png",format="PNG")
start=time.time()
a=readAllImgInFolder("dataset/pins_Adriana Lima")
b=getMean(a)
c=getDifference(a,b)
print(len(c),len(c[0]))
d=getCovariance(c)
print(len(d),len(d[0]))
q,r=eigen(d,50)
print(len(r),len(r[0]))
m,n=np.linalg.eig(d)
x=eigenFace(r,c)
saveEigenFaces(x)
# print("eigen value: ",q)
# print("eigen vector: ",r)
end=time.time()
# np.savetxt("eigenval.txt",q)
# np.savetxt("eigenvec.txt",r)
# np.savetxt("eigenvalnp.txt",m)
# np.savetxt("eigenvecnp.txt",n)
print((end-start)*10**3)