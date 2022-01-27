############################################################
# CIS 521: Individual Functions for CNN
############################################################

student_name = "Brian Grimaldi"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

from numpy.lib.arraypad import pad

############################################################
# Individual Functions
############################################################

def convolve_greyscale(image, kernel):
    
    kernelT = np.flip(kernel,(0,1))
    print(image.shape)
    print(kernelT.shape)
   
    row,col = image.shape
    kerR, kerC = kernel.shape
    new_image = np.zeros(image.shape)
    padded_image = image.copy()
    pRow=int((kerR-1)/2)
    pCol= int((kerC-1)/2)

    for i in range(pRow):
        padded_image = np.insert(padded_image,row,0,axis = 0)
        padded_image = np.insert(padded_image,0,0, axis = 0)

    for i in range(pCol):
        padded_image = np.insert(padded_image,col,0,axis = 1)
        padded_image = np.insert(padded_image,0,0, axis = 1)

    print(padded_image)

    row,col = image.shape
    for i in range(row):
        for j in range(col):
            sum = 0
            #Applying kernel filter
            for r in range(kerR):
                for c in range(kerC):
                    sum += padded_image[i+r,j+c]*kernelT[r,c]

            new_image[i,j] = sum

    return new_image


def convolve_rgb(image, kernel):
    
    new_image = np.zeros(image.shape)

    for i in range(3):
        new_image[:,:,i] = convolve_greyscale(image[:,:,i],kernel)

    return new_image

def max_pooling(image, kernel, stride):
    arr = []
    max_val = 0
    x_ctr = 0
    y_ctr = 0
    done = False
    rows, cols = image.shape
    row_arr = []
    while done != True:
        max_val = 0

        for kerR in range(kernel[0]):
            for kerC in range(kernel[1]):
                if image[x_ctr + kerR, y_ctr + kerC] > max_val:
                    max_val = image[x_ctr + kerR, y_ctr + kerC]
        row_arr.append(max_val)
        
        if x_ctr + kernel[0] + stride[0] > cols:
            x_ctr = 0
            arr.append(row_arr)
            if y_ctr +kernel[1] + stride[1] > rows: #Reaches end of max f(x)
                done = True #Exit while loop and return
            else:
                row_arr = []
                y_ctr += stride[1]
        else:
            x_ctr += stride[0]
    num_arr = np.array(arr)
    return num_arr.T

            


def average_pooling(image, kernel, stride):
    arr = []
    total = kernel[0]*kernel[1]
    x_ctr = 0
    y_ctr = 0
    done = False
    rows, cols = image.shape
    row_arr = []
    while done != True:
        avr_val = 0

        for kerR in range(kernel[0]):
            for kerC in range(kernel[1]):
                avr_val += image[x_ctr + kerR, y_ctr + kerC]

        row_arr.append(avr_val/total)
        
        if x_ctr + kernel[0] + stride[0] > cols:
            x_ctr = 0
            arr.append(row_arr)
            if y_ctr +kernel[1] + stride[1] > rows: #Reaches end of max f(x)
                done = True #Exit while loop and return
            else:
                row_arr = []
                y_ctr += stride[1]
        else:
            x_ctr += stride[0]
    num_arr = np.array(arr)
    return num_arr.T


def sigmoid(x):
    print(x)
    sig_list = [1/(1+math.exp(-xi)) for xi in x.flatten()]
    sig_list = (np.array(sig_list)).reshape(x.shape)
    return sig_list
# def main():

# #     #    image = np.array(Image.open('5.1.09.tiff'))
# #     #    plt.imshow(image, cmap='gray')
# #     #    plt.show()
# #     #    kernel_size = (4, 4)
# #     #    stride = (1, 1)
# #     #    output = average_pooling(image, kernel_size, stride)
# #     #    plt.imshow(output, cmap='gray')
# #     #    plt.show()
# #     #    print(output)
# #        x = np.array([0.5, 3, 1.5, -4.7, -100])
# #        print(sigmoid(x))
#     image = np.array([
#         [0, 1, -1],
#         [2, 1, 0],
#         [0, 3, -1]])

#     kernel = np.array([
#         [-1, 0, 1]])
    

#     print(convolve_greyscale(image, kernel))

# main()