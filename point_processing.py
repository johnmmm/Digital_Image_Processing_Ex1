import cv2 as cv  
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

input_url = '/Users/mac/Desktop/university/CST/1718Spring/6_数字图像处理/hw/1/git/Digital_Image_Processing_Ex1/src/input'
output_url = '/Users/mac/Desktop/university/CST/1718Spring/6_数字图像处理/hw/1/git/Digital_Image_Processing_Ex1/src/output'

def minone(x, y):
    if x > y:
        return y
    return x

def maxone(x, y):
    if x > y:
        return x
    return y

def brightness(inputimage, g = 5):
    outputimage = inputimage.copy()
    for i in range(0, len(inputimage)):
        for j in range(0, len(inputimage[0])):
            for k in range(0, 3):
                if inputimage[i][j][k] + g < 256 and inputimage[i][j][k] + g >= 0:
                    outputimage[i][j][k] = inputimage[i][j][k] + g
                elif inputimage[i][j][k] + g > 255:
                    outputimage[i][j][k] = 255
                elif inputimage[i][j][k] + g < 0:
                    outputimage[i][j][k] = 0
    return outputimage

def contrast(inputimage, a):
    outputimage = inputimage.copy()
    if a < 0:
        print('wrong number!!!\n')
        return outputimage
    else:
        for i in range(0, len(inputimage)):
            for j in range(0, len(inputimage[0])):
                for k in range(0, 3):
                    tmp_color = a * (inputimage[i][j][k] - 127) + 127
                    if tmp_color < 0:
                        outputimage[i][j][k] = 0
                    elif tmp_color >= 0 and tmp_color <= 255:
                        outputimage[i][j][k] = tmp_color
                    elif tmp_color > 255:
                        outputimage[i][j][k] = 255
    return outputimage

def gamma(inputimage, r):
    outputimage = inputimage.copy()
    if r < 0:
        print('wrong number!!!\n')
        return outputimage
    else:
        outputimage = 255 * (inputimage / 255) ** (1 / r)
    return outputimage

def histogram_equalization(inputimage):
    outputimage = inputimage.copy()
    total_size = len(inputimage) * len(inputimage[0])
    histo = {}#total
    for i in range(0, 256):
        histo[i] = 0
    for i in range(0, len(inputimage)):
        for j in range(0, len(inputimage[0])):
            histo[int(inputimage[i][j].sum() / 3)] += 1
    tmp = 0
    for j in range(0, 256):
        tmp += histo[j]
        histo[j] = tmp / total_size
    # building the histo
    for i in range(0, len(inputimage)):
        for j in range(0, len(inputimage[0])):
            for k in range(0, 3):
                outputimage[i][j][k] = 255 * histo[inputimage[i][j][k]]
    return outputimage

def histogram_matching(inputimage, matchingimage):
    outputimage = inputimage.copy()
    total_size1 = len(inputimage) * len(inputimage[0])
    total_size2 = len(matchingimage) * len(matchingimage[0])
    histo1 = {}
    histo2 = {}
    #for 1
    for i in range(0, 256):
        histo1[i] = 0
    for i in range(0, len(inputimage)):
        for j in range(0, len(inputimage[0])):
            histo1[int(inputimage[i][j].sum() / 3)] += 1
    tmp = 0
    for j in range(0, 256):
        tmp += histo1[j]
        histo1[j] = tmp / total_size1
    #for 2
    for i in range(0, 256):
        histo2[i] = 0
    for i in range(0, len(matchingimage)):
        for j in range(0, len(matchingimage[0])):
            histo2[int(matchingimage[i][j].sum() / 3)] += 1
    tmp = 0
    for j in range(0, 256):
        tmp += histo2[j]
        histo2[j] = tmp / total_size2

    for i in range(0, len(inputimage)):
        for j in range(0, len(inputimage[0])):
            gi = (int)(inputimage[i][j].sum() / 3)
            if gi == 0 or gi == 255:
                outputimage[i][j] = inputimage[i][j]
            else:
                histo_place = 0
                for gj in range(1, 255):
                    if histo2[gj] >= histo1[gi] and histo2[gj-1] < histo1[gi]:
                        histo_place = gj
                        break
                outputimage[i][j][0] = minone(255, (inputimage[i][j][0] / inputimage[i][j].sum()) * histo_place * 3) 
                outputimage[i][j][1] = minone(255, (inputimage[i][j][1] / inputimage[i][j].sum()) * histo_place * 3)
                outputimage[i][j][2] = minone(255, (inputimage[i][j][2] / inputimage[i][j].sum()) * histo_place * 3)
    return outputimage

def saturation(inputimage):
    #???
    return 0

def point_processing():
    image1 = cv.imread(input_url + '/4.jpg')
    # output_brigtness1 = brightness(image1, 50)
    # cv.imwrite(output_url + '/4_output_brigtness1.jpg', output_brigtness1)
    # output_brigtness2 = brightness(image1, -50)
    # cv.imwrite(output_url + '/4_output_brigtness2.jpg', output_brigtness2)
    # output_contrast1 = contrast(image1, 0.75)
    # cv.imwrite(output_url + '/4_output_contrast1.jpg', output_contrast1)
    # output_contrast2 = contrast(image1, 1.25)
    # cv.imwrite(output_url + '/4_output_contrast2.jpg', output_contrast2)
    # output_gamma1 = gamma(image1, 0.75)
    # cv.imwrite(output_url + '/4_output_gamma1.jpg', output_gamma1)
    # output_gamma2 = gamma(image1, 1.25)
    # cv.imwrite(output_url + '/4_output_gamma2.jpg', output_gamma2)
    # output_histogram_equalization = histogram_equalization(image1)
    # cv.imwrite(output_url + '/4_output_histogram_equalization.jpg', output_histogram_equalization)
    # image1 = cv.imread(input_url + '/7.png')
    # image2 = cv.imread(input_url + '/9.jpg')
    # output_histogram_matching = histogram_matching(image1, image2)
    # cv.imwrite(output_url + '/4_output_histogram_matching.jpg', output_histogram_matching)
    # ~~~~~~~~~~~~
    # cv.namedWindow('output1')
    # cv.imshow('output1', output_brigtness2)
    # cv.waitKey(0)
    return 0

point_processing()