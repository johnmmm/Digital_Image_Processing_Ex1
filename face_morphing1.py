import numpy as np
import dlib
import cv2 as cv
import sys
import scipy.signal
import matplotlib as plt


input_url = "./src/input/face_morphing"
output_url = "./src/output/face_morphing"

#有用的工具
def minone(x, y):
    if x > y:
        return y
    return x

def maxone(x, y):
    if x > y:
        return x
    return y

def mark_points(img, points):
    for point in points:
        img[point[1]][point[0]][0] = 255
        img[point[1]][point[0]][1] = 0
        img[point[1]][point[0]][2] = 0
        img[point[1]+1][point[0]][0] = 255
        img[point[1]+1][point[0]][1] = 0
        img[point[1]+1][point[0]][2] = 0
        img[point[1]-1][point[0]][0] = 255
        img[point[1]-1][point[0]][1] = 0
        img[point[1]-1][point[0]][2] = 0
        img[point[1]][point[0]+1][0] = 255
        img[point[1]][point[0]+1][1] = 0
        img[point[1]][point[0]+1][2] = 0
        img[point[1]][point[0]-1][0] = 255
        img[point[1]][point[0]-1][1] = 0
        img[point[1]][point[0]-1][2] = 0
        img[point[1]+1][point[0]+1][0] = 255
        img[point[1]+1][point[0]+1][1] = 0
        img[point[1]+1][point[0]+1][2] = 0
        img[point[1]+1][point[0]-1][0] = 255
        img[point[1]+1][point[0]-1][1] = 0
        img[point[1]+1][point[0]-1][2] = 0
        img[point[1]-1][point[0]+1][0] = 255
        img[point[1]-1][point[0]+1][1] = 0
        img[point[1]-1][point[0]+1][2] = 0
        img[point[1]-1][point[0]-1][0] = 255
        img[point[1]-1][point[0]-1][1] = 0
        img[point[1]-1][point[0]-1][2] = 0
    return img

def is_contain(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def get_point_index(points, points_array):
    for i in range(0, len(points_array)):
        if points_array[i][0] == int(points[0]) and points_array[i][1] == int(points[1]):
            return i
    return -1

def bounding_rect(points):
    maxx, maxy = 0, 0
    minx, miny = 10000, 10000
    for i in points:
        if i[0] > maxx:
            maxx = i[0]
        if i[0] < minx:
            minx = i[0]
        if i[1] > maxy:
            maxy = i[1]
        if i[1] < miny:
            miny = i[1]
    return [minx, miny, maxx - minx, maxy - miny]

def cat_img(img1, ratio1, img2, ratio2):
    img3 = img1 * ratio1 + img2 * ratio2
    return img3

def get_feature_points(img, dlibDector, dlibPredictor):
    dets = dlibDector(img, 1)
    if len(dets) != 1:
        print ("Face number is not 1!!!")
        return []
    
    shape = dlibPredictor(img, dets[0])
    keyPoints = []
    for i in range(0, shape.num_parts):
        keyPoints.append((shape.part(i).x, shape.part(i).y))
    imgshape = img.shape
    keyPoints.append((0, 0))
    keyPoints.append((0, imgshape[0] - 1))
    keyPoints.append((0, (imgshape[0] - 1) / 2))
    keyPoints.append(((imgshape[1] - 1) / 2, 0))
    keyPoints.append(((imgshape[1] - 1) / 2, imgshape[0] - 1))
    keyPoints.append((imgshape[1] - 1, 0))
    keyPoints.append((imgshape[1] - 1, (imgshape[0] - 1) / 2))
    keyPoints.append((imgshape[1] - 1, imgshape[0] - 1))
    return keyPoints

#face_morphing
def fill_morph(img_target, img_source, triangle1, triangle2):
    img_tmp = np.zeros_like(img_source)
    rect1 = cv.boundingRect(triangle1)
    rect2 = cv.boundingRect(triangle2)

    new_tri1 = []
    new_tri2 = []

    for i in range(0, 3):
        new_tri1.append(((triangle1[0][i][0] - rect1[0]), (triangle1[0][i][1] - rect1[1])))
        new_tri2.append(((triangle2[0][i][0] - rect2[0]), (triangle2[0][i][1] - rect2[1])))
    new_img1 = img_source[rect1[1] : rect1[1] + rect1[3], rect1[0] : rect1[0] + rect1[2]]
    warp_mat = cv.getAffineTransform(np.float32(new_tri1), np.float32(new_tri2))
    new_img2 = cv.warpAffine(new_img1, warp_mat, (rect2[2], rect2[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    mask = np.zeros((rect2[3], rect2[2], 3), dtype = np.float32)
    cv.fillConvexPoly(mask, np.int32(new_tri2), (1.0, 1.0, 1.0), 16, 0)#三角掩码
    new_img2 = new_img2 * mask

    # print(rect2)
    # print(len(img_tmp[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]][0]))
    img_tmp[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img_tmp[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * ((1.0, 1.0, 1.0) - mask)
    img_tmp[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img_tmp[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] + new_img2

    print("fill morphing!!!")

    #为了寻找其中不是0的点
    img_transpose = np.transpose(np.nonzero(img_target * img_tmp))
    img_target = img_target + img_tmp
    for i in img_transpose:
        img_target[i[0]][i[1]] = (img_target[i[0]][i[1]] + img_tmp[i[0]][i[1]]) / 2

    return img_target

def face_morphing(img1, img2, points1, points2, ratio):
    points3 = points1 * ratio + points2 * (1 - ratio)
    points3 = points3.astype("uint32")
    for i in range(0, len(points3)):
        points3[i] = (points3[i, 0], points3[i, 1])

    imgshape2 = img2.shape
    rect2 = (0, 0, imgshape2[1], imgshape2[0])
    subdiv = cv.Subdiv2D(rect2)

    #print(len(points3))
    for i in points3:
        subdiv.insert((i[0], i[1]))
    triangles3 = subdiv.getTriangleList()
    triangle_id = []
    for i in triangles3:
        p1 = (i[0], i[1])
        p2 = (i[2], i[3])
        p3 = (i[4], i[5])
        #大图缩小，需要排除一些点
        if is_contain(rect2, p1) and is_contain(rect2, p2) and is_contain(rect2, p3):
            triangle_id.append([get_point_index(p1, points3), get_point_index(p2, points3), get_point_index(p3, points3)])

    # for i in range triangle_id:
    #     print(i)

    new_img1 = np.zeros_like(img1)
    new_img2 = np.zeros_like(img2)
    for i in range(0, len(triangle_id)):
        triangle1 = np.float32([[ [points1[triangle_id[i][0]][0], points1[triangle_id[i][0]][1]],
                                  [points1[triangle_id[i][1]][0], points1[triangle_id[i][1]][1]],
                                  [points1[triangle_id[i][2]][0], points1[triangle_id[i][2]][1]]]])
        triangle2 = np.float32([[ [points2[triangle_id[i][0]][0], points2[triangle_id[i][0]][1]],
                                  [points2[triangle_id[i][1]][0], points2[triangle_id[i][1]][1]],
                                  [points2[triangle_id[i][2]][0], points2[triangle_id[i][2]][1]]]])
        triangle3 = np.float32([[ [points3[triangle_id[i][0]][0], points3[triangle_id[i][0]][1]],
                                  [points3[triangle_id[i][1]][0], points3[triangle_id[i][1]][1]],
                                  [points3[triangle_id[i][2]][0], points3[triangle_id[i][2]][1]]]])

        new_img1 = fill_morph(new_img1, img1, triangle1, triangle3)
        new_img2 = fill_morph(new_img2, img2, triangle2, triangle3)

    result = cat_img(new_img1, ratio, new_img2, 1-ratio)
    return result

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_path1 = input_url + "/hilary.png"
face_path2 = input_url + "/cruz.png"
save_path = output_url + "/test.png"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

img1 = cv.imread(face_path1)
img2 = cv.imread(face_path2)

#稍微裁剪一下，让两个图一样大
img_row = minone(len(img1), len(img2))
img_col = minone(len(img1[0]), len(img2[0]))
img1 = img1[0:img_row, 0:img_col]
img2 = img2[0:img_row, 0:img_col]

points1 = get_feature_points(img1, face_detector, shape_predictor)
points2 = get_feature_points(img2, face_detector, shape_predictor)

#获取失败
if len(points1) == 0 or len(points2) == 0:
    print("ERROR!!!")
    exit()

points_array1 = np.array(points1)
points_array2 = np.array(points2)

img3 = face_morphing(img1, img2, points_array1, points_array2, 0.5)

#标记特征点
#img = mark_points(img, points)

cv.imwrite(save_path, img3 )