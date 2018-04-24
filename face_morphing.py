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

def face_morphing():
    
    return 0