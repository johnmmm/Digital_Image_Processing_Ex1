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

def image_fusion():
    #image_fusion
    pic_url = '/image_fusion'
    image_src1 = cv.imread(input_url + pic_url + '/test1_src.jpg')
    image_target1 = cv.imread(input_url + pic_url + '/test1_target.jpg')
    image_mask1 = cv.imread(input_url + pic_url + '/test1_mask.jpg')

    image_src2 = cv.imread(input_url + pic_url + '/test2_src.png')
    image_target2 = cv.imread(input_url + pic_url + '/test2_target.png')
    image_mask2 = cv.imread(input_url + pic_url + '/test2_mask.png')

    #matlab xiefa
    image_mask2_bw = np.zeros((len(image_mask2), len(image_mask2[0])))
    for i in range(0, len(image_mask2)):
        for j in range(0, len(image_mask2[0])):
            if image_mask2[i][j][0] > 100:
                image_mask2_bw[i][j] = 1
            elif image_mask2[i][j][0] < 100:
                image_mask2_bw[i][j] = 0
    
    image_src2_b = np.zeros((len(image_src2), len(image_src2[0])))
    image_src2_g = np.zeros((len(image_src2), len(image_src2[0])))
    image_src2_r = np.zeros((len(image_src2), len(image_src2[0])))
    for i in range(0, len(image_src2_b)):
        for j in range(0, len(image_src2_b[0])):
            image_src2_b[i][j] = image_src2[i][j][0]
            image_src2_g[i][j] = image_src2[i][j][1]
            image_src2_r[i][j] = image_src2[i][j][2]

    image_target2_b = np.zeros((len(image_target2), len(image_target2[0])))
    image_target2_g = np.zeros((len(image_target2), len(image_target2[0])))
    image_target2_r = np.zeros((len(image_target2), len(image_target2[0])))
    for i in range(0, len(image_target2)):
        for j in range(0, len(image_target2[0])):
            image_target2_b[i][j] = image_target2[i][j][0]
            image_target2_g[i][j] = image_target2[i][j][1]
            image_target2_r[i][j] = image_target2[i][j][2]

    image_src2_b = image_src2_b * image_mask2_bw
    image_src2_g = image_src2_g * image_mask2_bw
    image_src2_r = image_src2_r * image_mask2_bw
    
    image_ans2_b = poisson_calculate(image_target2_b, 152, 145, image_mask2_bw, image_src2_b)
    image_ans2_g = poisson_calculate(image_target2_g, 152, 145, image_mask2_bw, image_src2_g)
    image_ans2_r = poisson_calculate(image_target2_r, 152, 145, image_mask2_bw, image_src2_r)

    for i in range(0, len(image_target2)):
        for j in range(0, len(image_target2[0])):
            image_target2[i][j][0] = image_ans2_b[i][j]
            image_target2[i][j][1] = image_ans2_g[i][j]
            image_target2[i][j][2] = image_ans2_r[i][j]
    cv.imwrite(output_url + '/image_fusion_2.jpg', image_target2)

    return 0

def poisson_calculate(targetimg, target_x, target_y, sourcemask, sourceimg):
    #sourcemask from 0 to 1
    #sourceimg为在这个source之中那些为0，哪些为1
    max_height, min_height, max_width, min_width = 0, len(sourceimg), 0, len(sourceimg[0])
    for i in range(0, len(sourceimg)):
        for j in range(0, len(sourceimg[0])):
            if i > max_height:
                max_height = i
            if i < min_height:
                min_height = i
            if j > max_width:
                max_width = j
            if j < min_width:
                min_width = j

    #重新建立一个
    source_object = np.zeros((max_height-min_height+1+2, max_width-min_width+1+2))
    source_mask = np.zeros((max_height-min_height+1+2, max_width-min_width+1+2))
    for i in range(min_height, max_height+1):
        for j in range(min_width, max_width+1):
            source_object[i-min_height+1+1, j-min_width+1+1] = sourceimg[i][j]
            source_mask[i-min_height+1+1, j-min_width+1+1] = sourcemask[i][j]

    #make boundary filter
    source_row, source_col = len(source_object), len(source_object[0])
    boundary_filter = np.zeros((source_row, source_col))
    boundary_pixel = np.zeros((source_row, source_col))
    for i in range(0, source_row):
        for j in range(0, source_col):
            if boundary_test(source_mask, i, j) == 1:
                boundary_filter[i, j] = 1
                boundary_pixel[i, j] = targetimg[target_x+i][target_y+j]

    gradience = np.zeros((source_row, source_col))
    gradience_filter = np.zeros((3, 3))
    gradience_filter = [[0, -1, 0], 
                        [-1, 4, -1],
                        [0, -1, 0]]
    #提前计算好梯度，作为后面方程中的b
    gradience = scipy.signal.convolve2d(source_object, gradience_filter, 'same')
    gradience = gradience * source_mask;
    gradience = gradience * (1 - boundary_filter)
    gradience = gradience + boundary_pixel;

    gauss_targetimg = boundary_pixel
    gauss_xs = source_mask - boundary_filter

    for i in range(0, 5000):
        if i % 200 == 0:
            print(i)
        for i in range(0, source_row):
            for j in range(0, source_col):
                if gauss_xs[i, j] > 0:
                    gauss_targetimg[i,j] = minone(1 / 4 * (gradience[i, j] + gauss_targetimg[i-1, j] + gauss_targetimg[i, j+1] + gauss_targetimg[i, j-1] + gauss_targetimg[i+1, j]), 255)
                    gauss_targetimg[i,j] = maxone(gauss_targetimg[i,j], 0)

    gauss_targetimg = gauss_targetimg * source_mask
    for i in range(0, source_row):
        for j in range(0, source_col):
            if gauss_targetimg[i, j] != 0:
                targetimg[target_x+i][target_y+j] = gauss_targetimg[i, j]

    for i in range(0, source_row):
        for j in range(0, source_col):
            targetimg[target_x+i][target_y+j] = minone(targetimg[target_x+i][target_y+j], 255)
            targetimg[target_x+i][target_y+j] = maxone(targetimg[target_x+i][target_y+j], 0)

    return targetimg

def boundary_test(inputimage, i, j):
    source_row, source_col = len(inputimage), len(inputimage[0])
    if i - 1 >= 1 and i + 1 <= source_row  and  j - 1 >= 1 and j + 1 <= source_col:
        if inputimage[i][j] != 0:
            if inputimage[i+1][j] == 0 or inputimage[i-1][j] == 0 or inputimage[i][j+1] == 0 or inputimage[i][j-1] == 0 or inputimage[i-1][j-1] == 0 or inputimage[i-1][j+1] == 0 or inputimage[i+1][j-1] == 0 or inputimage[i+1][j+1] == 0:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

image_fusion()