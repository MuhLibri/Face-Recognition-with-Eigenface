import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk,Image
from faceRecog import *
from tesss import *
from scipy import interpolate
import time
import  numpy as np

imgfiletype=[("jpg image","*.jpg"),("png image","*.png")]

global foldir
def getFoldir():
    global foldir
    foldir=filedialog.askdirectory()
    l1.config(text=foldir)
global imgdir
def getImgdir():
    global imgdir
    imgdir=filedialog.askopenfilename(filetypes=imgfiletype)
    l2.config(text=imgdir)
    print(imgdir)
    newImg=ImageTk.PhotoImage(Image.open(imgdir).resize((256,256)))
    limg1.configure(image=newImg)
    limg1.image=newImg
    # canvas1.itemconfig(canvas2,image=ImageTk.PhotoImage(Image.open(imgdir)))
def processimg():
    global imgdir
    start=time.time()
    a = readAllImgInFolder(foldir)
    a2=readAllImgInFolderRGB(foldir)
    img = cv2.imread(imgdir,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    if img is not None:
        img = np.reshape(img, -1)
    print("1")
    b = getMean(a)
    print("2")
    c = getDifference(a, b)
    print("3")
    d = getCovariance(c)
    print("4")
    val, vec = eigen(d, 60)
    # np.savetxt("eigenval.txt",val)
    # np.savetxt("eigenvalnp.txt",np.linalg.eig(d)[0])
    print("5")
    # val1, vec1 = np.linalg.eig(d)
    x = eigenFace(vec, c)
    # saveEigenFaces(x)
    print("6")
    # x2 = eigenFace(vec1, c)
    # m = interpolate.interp1d([min(img), max(img)], [0, 255])
    img=[img]
    i = facerecog(img, x,c,b)
    # pp, ii = facerecog(m(img), x2)
    # print(i)
    end=time.time()
    if(i>=0):
        img2=cv2.cvtColor(a2[i],cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)
        # newImg = ImageTk.PhotoImage(img2.resize((256, 256)))
        newImg = ImageTk.PhotoImage(img2)
    else:
        newImg=ImageTk.PhotoImage(Image.open("notfound.jpg").resize((256, 256)))

    limg2.configure(image=newImg)
    limg2.image=newImg
    lextime2.configure(text=str(end-start))
window = tk.Tk()
window.geometry("720x480")
# greeting = tk.Label(text="Hello, Tkinter")
# greeting.pack()

l1=tk.Label(window,text="no file selected",bg="lightblue",width=60,height=1)
l2=tk.Label(window,text="no file selected",bg="yellow",width=60,height=1)
l1.grid(row=0,column=1,sticky='w',columnspan=5)
l2.grid(row=1,column=1,sticky='w',columnspan=5)

# canvas1 = tk.Canvas(window,width=300,height=300)
# canvas1.grid(row=1,column=2)
# canvas2 = tk.Canvas(window,width=300,height=300)
# canvas2.grid(row=2,column=3)
img1=ImageTk.PhotoImage(Image.open("defaultImg.jpg").resize((256,256)))
# canvas1.create_image(20,20,image=img1)
img2=ImageTk.PhotoImage(Image.open("defaultImg.jpg").resize((256,256)))
# canvas2.create_image(20,20,image=img2)
limg1=tk.Label(window)
limg1.grid(row=2,column=0,sticky='w',columnspan=3,padx=10,pady=10)
limg1.config(width=300,height=300)
limg2=tk.Label(window)
limg2.grid(row=2,column=3,sticky='w',columnspan=3,padx=10,pady=10)
limg2.config(width=300,height=300)
limg1.config(image=img1)
limg2.config(image=img2)
lextime1=tk.Label(window,text="execution time:",width=15,height=1)
lextime1.grid(row=3,column=4)
lextime2=tk.Label(window,text="",width=15,height=1)
lextime2.grid(row=3,column=5)

button1=tk.Button(window,text="input folder",bg="red",justify="left")
button2=tk.Button(window,text="input image",bg="red",justify="left")
button1.config(command=lambda:getFoldir())
button2.config(command=lambda:getImgdir())
button1.grid(row=0,column=0,padx=5,pady=5)
button2.grid(row=1,column=0,padx=5,pady=5)
processbutton=tk.Button(window,text="process image",bg="red",justify="center")
processbutton.config(command=lambda:processimg())
processbutton.grid(row=3,column=2,sticky="nwse",columnspan=2)
n_rows=4
n_columns=6
for i in range(n_rows):
    window.grid_rowconfigure(i,  weight =1)
for i in range(n_columns):
    window.grid_columnconfigure(i,  weight =1)

window.mainloop()