import os
import time
import random
import PIL.Image
from cv2 import  *
import cv2
from PIL import *
import  numpy as np
import scipy
def readAllImgInFolderRGB(path):
    imglist=[]
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path,file),cv2.IMREAD_ANYCOLOR)
        # img = cv2.resize(img,(256,256))
        if img is not None:
            # img=np.reshape(img,-1)
            imglist.append(img)
    return imglist
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
    sum=np.divide(sum,l)
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
    # at=np.transpose(a)
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
# def QRDecomposition(covariant):
#     A=np.copy(covariant)
#     # print(len(A),len(A[0]))
#
#     n,m=covariant.shape
#     # n-=2
#     # m-=2
#     d=np.zeros_like(covariant[0])
#     # A=np.copy(covariant)
#     for i in range (n):
#         s=np.linalg.norm(covariant[i:m,i])
#         if(covariant[i,i]>=0):
#             d[i]=-s
#         else:
#             d[i]=s
#         fak=np.sqrt(s*(s+abs(covariant[i,i])))
#         A[i,i]=A[i,i]-d[i]
#         A[i:m,i]=A[i:m,i]/fak
#         # print(d[i])
#         if(i<n):
#             # print(A[i:m,i].shape)
#             # print(np.transpose(A[i:m,i]).shape)
#             # print(A[i:m,i:n].shape)
#             A[i:m,i+1:n]=A[i:m,i+1:n]-np.matmul(A[i:m,i],(np.matmul(np.transpose(A[i:m,i]),A[i:m,i:n])))
#     # print(A)
#     return A,d

'''d=np.zeros_like(covariant[0])
    # A=np.copy(covariant)
    for i in range (n):
        s=np.linalg.norm(covariant[i:m])
        if(covariant[i][i]>=0):
            d[i]=-s
        else:
            d[i]=s
        fak=np.sqrt(s*(s+abs(covariant[i][i])))
        A[i][i]=A[i][i]-d[i]
        A[i:m]=A[i:m]/fak
        # print(d[i])
        if(i<n):

            A[i]=A[i]-np.transpose(A[i])*A[i]*A[i]
    # print(A)
    return A,d'''

'''A=np.array(covariant)
    n,m=covariant.shape
    n-=1
    m-=1
    d=np.zeros_like(covariant[0])
    # A=np.copy(covariant)
    for i in range (n):
        s=np.linalg.norm(covariant[i:m][i])
        if(covariant[i][i]>=0):
            d[i]=-s
        else:
            d[i]=s
        fak=np.sqrt(s*(s+abs(covariant[i][i])))
        A[i][i]=A[i][i]-d[i]
        A[i:m][i]=A[i:m][i]/fak
        if(i<n):
            A[i:m][(i+1):n]=A[i:m][(i+1):n]-A[i:m][i]*np.transpose(A[i:m][i])*A[i:m][(i+1):n]
    return A,d
'''
def QRgram(A):
    # z=[]
    m,n=A.shape
    R=np.zeros_like(A)
    Q=np.copy(A)
    for k in range (n):
        t=np.linalg.norm(Q[:,k])
        # t=np.sqrt(np.sum(np.abs(Q[:,k]) ** 2))
        nach=True
        u=0
        while nach:
            u+=1
            for i in range(k-1):
                s=np.matmul(np.transpose(Q[:,i]),Q[:,k])
                R[i,k]=R[i,k]+s
                Q[:,k]=Q[:,k]-s*Q[:,i]
            tt=np.linalg.norm(Q[:,k])
            # tt=np.sqrt(np.sum(np.abs(Q[:,k]) ** 2))
            if (tt>10*(2.2204e-16)*t and tt<t/10):
                nach=True
                t=tt
            else:
                nach=False
                if tt < 10 * (2.2204e-16) * t:
                    tt=0
        # z.append(u)
        R[k,k]=tt
        if ((tt*(2.2204e-16)) !=0):
            tt=1/tt
        else:
            tt=0
        Q[:,k]=Q[:,k]*tt
    # print(len(Q),len(Q[0]))
    # print(len(R), len(R[0]))
    return Q,R

def QRclassicalSchmidt(A):
    m,n=A.shape
    R=np.zeros_like(A)
    Q=np.copy(A)
    for k in range (n):
        for i in range (1,k-1):
            R[i,k]=np.matmul(np.transpose(Q[:,i]),Q[:,k])
        for i in range (1,k-1):
            Q[:,k]=np.subtract(Q[:,k],np.matmul(R[i,k],Q[:,i]))
        R[k,k]=np.linalg.norm(Q[:,k])
        Q[:,k]=np.divide(Q[:,k],R[k,k])
    return  Q,R

def QRgrammod(B):
    A = np.copy(B)
    m,n=A.shape
    Q=np.zeros_like(A)
    R=np.zeros_like(A)
    for k in range (n):
        s=0
        for j in range (m):
            s=s+np.power(A[j,k],2)
        R[k,k]=np.sqrt(s)
        for j in range (m):
            Q[j,k]=np.divide(A[j,k],R[k,k])
        for i in range (k+1,n):
            s=0
            for j in range (m):
                s=s+A[j,i]*Q[j,k]
            R[k,i]=s
            for j in range (m):
                A[j,i]=A[j,i]-R[k,i]*Q[j,k]
    return Q,R
# def QR(A):
#     n,m=A.shape
#     Q=np.empty((n,n))
#     u=np.empty((n,n))
#     u[:,0]=A[:,0]
#     Q[:,0]=u[:,0]/np.linalg.norm(u[:,0])
#     for i in range

def eigen(A,maxiter=1000):
    n=len(A)
    U = np.identity(n)
    eigenV=[]
    for i in range(maxiter):
        # Q,R = QR_factorisation_Householder_double(A)
        # Q,R=QRgram(A)
        # Q,R=QRclassicalSchmidt(A)
        Q,R=QRgrammod(A)
        # print(R)
        # print(R)
        print(i)
        A=np.matmul(R,Q)
        U=np.dot(U,Q)
    for vec in U:
        eigenV.append(vec)
    eigval=np.diag(A)
    # pel=list(zip(eigval,eigenV))
    # print(pel)
    # for i in range(len(eigval)):
    #     pel.append((eigval[i],eigenV[i]))
    # pel.sort(key=lambda tup: tup[0], reverse=True)
    # eigvals=[]
    # eigvecs=[]
    # for x in pel:
    #     eigval,eigvec=x
    #     eigvals.append(eigval)
        # eigvecs.append(eigvec)
    # print(A)
    # print(len(A),len(A[0]))
    return eigval,eigenV;
# def eigen(A, maxiter=1000):
#     n = len(A)
#     U = np.identity(n)
#     eigenV = []
#     for i in range(maxiter):
#         pel = []
#         A, D = QRDecomposition(A)
#         # A = np.matmul(R, Q)
#         # U = np.matmul(U, Q)
#     for vec in A:
#         eigenV.append(vec)
#     return np.diag(A), eigenV;
def eigenFace(eigenV,difference):
    eigenfaces=[]

    for i in range (len(difference)):
        p=np.dot(eigenV[i],difference)
        m = scipy.interpolate.interp1d([min(p), max(p)], [0, 255])
        eigenfaces.append(m(p))
        # print(len(eigenfaces[i]))
    return eigenfaces
def saveEigenFaces(eigenfaces):
    print(len(eigenfaces[0]))
    for i in range(len(eigenfaces)):
        img=PIL.Image.fromarray(eigenfaces[i].reshape(256,256))
        img=img.convert("L")
# a=[[random.random() for i in range(3)] for i in range(3)]
# print(a)
# Qgr,Rgr=QRgrammod(np.array(a))
# qh,rh=QR_factorisation_Householder_double(a)
# print("r:")
# print(Rgr)
# print(rh)
# print("q:")
# print(Qgr)
# print(qh)
# print("a:")
# print(np.matmul(Qgr,Rgr))
# print(np.matmul(qh,rh))
# print("npqr:")
# print(np.linalg.qr(a)[0])
# print(np.linalg.qr(a)[1])
# A=np.copy(a)
# B=np.copy(a)
# for i in range (1):
#     Qgr,Rgr=QRgram(A)
#     qh,rh=QR_factorisation_Householder_double(B)
#     A=np.matmul(Rgr,Qgr)
#     B=np.matmul(rh,qh)
# print(Rgr)
# print(rh)
# np.savetxt("qrgram.txt",np.matmul(Rgr,Qgr))
# np.savetxt("househ.txt",np.matmul(rh,qh))
# a=readAllImgInFolder("dataset/all_12")
# b=getMean(a)
# c=getDifference(a,b)
# d=getCovariance(c)
# val,vec=eigen(d,1)
# # val,vec=np.linalg.eig(d)
# x=eigenFace(vec,c)
# saveEigenFaces(x)
# m,n=np.linalg.eig(d)
# np.savetxt("eigenval.txt",val)
# np.savetxt("eigenvec.txt",vec)
# np.savetxt("eigenvalnp.txt",m)
# np.savetxt("eigenvecnp.txt",n)
# end=time.time()
# print((end-start))