#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:47:04 2018

@author: raphaelchekroun
"""

import numpy as np 
# this is the key library for manipulating arrays. Use the online ressources! http://www.numpy.org/

import matplotlib.pyplot as plt 
# used to read images, display and plot http://matplotlib.org/api/pyplot_api.html . 
#You can also check this simple intro to using ipython notebook with images https://matplotlib.org/users/image_tutorial.html

# to display directly in the notebook

import scipy.ndimage as ndimage
# one of several python libraries for image procession

plt.rcParams['image.cmap'] = 'gray' 
# by default, the grayscale images are displayed with the jet colormap: use grayscale instead

##### I'm gonna use this function later #####
def display_image(I):
    plt.imshow(I)
    plt.show()

def RGB_to_grey(tuple):
    return np.dot(np.asarray(tuple),[0.2989,0.5870,0.1140])


def est_gris(I):
    return (len(I[0][0])==4)

def load_image(name,crop_window=-1): 
    I=plt.imread(name)
    if est_gris(I):
        print("La photo est déjà grise")
        return I
    M=np.zeros((I.shape[0],I.shape[1]))
    M=RGB_to_grey(I)
    if crop_window!=-1:
        M=M[crop_window[0]:crop_window[1],crop_window[2]:crop_window[3]]
    M=M.astype('float')/255
    return M

def compute_gradient(I,sigma=2):
    I_hor=ndimage.gaussian_filter(I,sigma=sigma,order=(1,0),mode='constant')
    I_ver=ndimage.gaussian_filter(I,sigma=sigma,order=(0,1),mode='constant')
    I_grad=np.sqrt(I_hor*I_hor + I_ver*I_ver)
    return I_hor,I_ver,I_grad


def round_angle(angle):
    if -9/8*np.pi <= angle < - 7/8*np.pi:
        return 0
    elif - 7/8*np.pi <= angle < -5/8*np.pi:
        return 1
    elif - 5/8*np.pi <= angle < -3/8*np.pi:
        return 2
    elif - 3/8*np.pi <= angle < -1/8*np.pi:
        return 3
    elif - 1/8*np.pi <= angle < 1/8*np.pi:
        return 0
    elif 1/8*np.pi <= angle < 3/8*np.pi:
        return 1
    elif 3/8*np.pi <= angle < 5/8*np.pi:
        return 2
    elif 5/8*np.pi <= angle < 7/8*np.pi:
        return 3      
    elif 7/8*np.pi <= angle < 9/8*np.pi:
        return 0



def nms(I, sigma):
    I_hor,I_ver,I_grad=compute_gradient(I,sigma)
    Dir=np.arctan2(I_hor,I_ver)
    line, column = I_grad.shape
    M = np.zeros((line,column), dtype=np.int32)
    for i in range(line-1):
        for j in range(column-1):
            where = round_angle(Dir[i, j])
            if where == 0:
                if (I_grad[i, j] >= I_grad[i, j - 1]) and (I_grad[i, j] >= I_grad[i, j + 1]):
                    M[i,j] = 1
            elif where == 1:
                if (I_grad[i, j] >= I_grad[i - 1, j]) and (I_grad[i, j] >= I_grad[i + 1, j]):
                    M[i,j] = 1
            elif where == 2:
                if (I_grad[i, j] >= I_grad[i - 1, j - 1]) and (I_grad[i, j] >= I_grad[i + 1, j + 1]):
                    M[i,j] = 1
            elif where == 3:
                if (I_grad[i, j] >= I_grad[i - 1, j + 1]) and (I_grad[i, j] >= I_grad[i + 1, j - 1]):
                    M[i,j] = 1
    return M,I_grad,Dir



def c(I,th,sigma):
    M,I_grad,_ = nms(I,sigma)
    n,p=I_grad.shape
    O=np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            if I_grad[i][j]<th:
                M[i][j]=0
    return M


def voisin_present(L_max,i,j):
    if [i-1,j] in L_max:
        return True
    elif [i-1,j-1] in L_max:
        return True
    elif [i-1,j+1] in L_max:
        return True
    elif [i,j-1] in L_max:
        return True
    elif [i,j+1] in L_max:
        return True
    elif [i+1,j] in L_max:
        return True
    elif [i+1,j-1] in L_max:
        return True
    elif [i+1,j+1] in L_max:
        return True
    else:
        return False
    

def canny_edge(I,sigma,thd,thu):
    M,I_grad,Dir=nms(I,sigma)
    n,p=I_grad.shape
    O=np.zeros((n,p))
    OK=[]
    V=[]
    for i in range(n):
        for j in range(p):
            if M[i][j]==1 and I_grad[i][j]>=thu:
                O[i][j]=1
                OK.append((i,j))
            elif M[i][j]==1 and thu>I_grad[i][j]>=thd:
                V.append((i,j))
    for (i,j) in OK:
        if (i-1,j-1) in V:
            O[i-1][j-1]=1
            OK.append((i-1,j-1))
            V.remove((i-1,j-1))
        if (i-1,j) in V:
            O[i-1][j]=1
            OK.append((i-1,j))
            V.remove((i-1,j))
        if (i-1,j+1) in V:
            O[i-1][j+1]=1
            OK.append((i-1,j+1))
            V.remove((i-1,j+1))
        if (i,j+1) in V:
            O[i][j+1]=1
            OK.append((i,j+1))
            V.remove((i,j+1))
        if (i,j-1) in V:
            O[i][j-1]=1
            OK.append((i,j-1))
            V.remove((i,j-1))
        if (i+1,j-1) in V:
            O[i+1][j-1]=1
            OK.append((i+1,j-1))
            V.remove((i+1,j-1))
        if (i+1,j) in V:
            O[i+1][j]=1
            OK.append((i+1,j))
            V.remove((i+1,j))
        if (i+1,j+1) in V:
            O[i+1][j+1]=1
            OK.append((i+1,j+1))
            V.remove((i+1,j+1))
    return O


I=load_image('lena.jpg')
display_image(I)
O=canny_edge(I,3,0.001,0.02)
display_image(O)
plt.imsave("canny_lena.jpg",O)

