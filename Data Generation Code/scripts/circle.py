import math
import random
# import shapely.geometry
from shapely.geometry import LineString, Point, MultiPoint
import cv2
import torch
import pickle
import open3d
import numpy as np
from math import sqrt

# (h, w) = img.shape[:2]
# h, w = 20,30
# center = (w / 2, h / 2)
# angle = 45
# scale = 1.0

# # 计算旋转矩阵
# M = cv2.getRotationMatrix2D(center, angle, scale)

def check_list(lst):
    # 定义三个状态：0表示初始状态，1表示第一部分负数，2表示中间部分正数，3表示最后一部分负数
    state = 0
    for num in lst:
        if state == 0:
            if num < 0:
                state = 1
            else:
                return False
        elif state == 1:
            if num > 0:
                state = 2
        elif state == 2:
            if num < 0:
                state = 3
        elif state == 3:
            if num > 0:
                return False
    return state == 3

def cvt_coords(coords, height):
    return [(x, height - y) for x, y in coords]

def intersection(rectangle, point1, point2, rectangle_height):
    # 创建矩形对象

    # 换算到图像所在坐标系
    quad = rectangle + [rectangle[0]]
    line = LineString([point1, point2])
    quad_cv = cvt_coords(quad, rectangle_height)
    line_cv = LineString(cvt_coords(line.coords[:], rectangle_height))

    quad_line = LineString(quad_cv)
    intersect = line_cv.intersection(quad_line)
    if isinstance(intersect, Point):
        return [list(intersect.coords)]
    elif isinstance(intersect, MultiPoint):
        return [list(p.coords) for p in intersect.geoms]
    else:
        return []




    # rect = LineString(rectangle + [rectangle[0]])
    # # 创建直线对象
    # line = LineString([point1, point2])
    # # 计算交点
    # inter = rect.intersection(line)
    # # 判断交点类型
    # if isinstance(inter, Point):
    #     return [list(inter.coords)]
    # elif isinstance(inter, MultiPoint):
    #     return [list(p.coords) for p in inter.geoms]
    # else:
    #     return []
def angle_with_x_axis(point1, point2):
    
    # check which point has a smaller x-coordinate
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    # if point1[0] > point2[0]:       
    #     x1, y1, x2, y2 = x2, y2, x1, y1
    # calculate the angle between the vector connecting the two points and the positive x-axis
    angle = math.atan2(y2 - y1, x2 - x1)

    # convert the angle to degrees
    angle_degrees = math.degrees(angle)
    # 返回结果
    return angle_degrees

def rectangle_circumcircle(vertices):
    # 计算矩形的中心点
    center_x = (vertices[0][0] + vertices[1][0] + vertices[2][0] + vertices[3][0]) / 4
    center_y = (vertices[0][1] + vertices[1][1] + vertices[2][1] + vertices[3][1]) / 4
    center = (center_x, center_y)

    rectangle_height = 2*center_y

    # 计算矩形对角线长度的一半，即为外接圆半径
    radius = math.sqrt((vertices[0][0] - center_x) ** 2 + (vertices[0][1] - center_y) ** 2)

    # 圆的方程
    equation = f'(x-{center_x})^2 + (y-{center_y})^2 = {radius**2}'

    # 在圆上随机采样两个点
    angle1 = random.uniform(0, 2 * math.pi)
    angle2 = random.uniform(0, 2 * math.pi)
    point1 = (center_x + radius * math.cos(angle1), center_y + radius * math.sin(angle1))
    point2 = (center_x + radius * math.cos(angle2), center_y + radius * math.sin(angle2))

    # 计算两个采样点之间的距离
    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # 计算两个采样点连成的直线的方程 y=k*x+b
    intersections = np.array(intersection(vertices, point1, point2,rectangle_height))
    intersections = np.squeeze(intersections)
    intersections_cv = cvt_coords(intersections, rectangle_height)
    # if intersections.size != 0:
    #     intersections_flip = np.fliplr(intersections)
    # point1, point2= (0,0), (2,4) # 63
    # point1, point2= (799,163), (0,808) # 141
    if intersections.size == 4:
        point1, point2= intersections_cv[0] ,intersections_cv[1] # 
        angle = angle_with_x_axis(point1, point2)
    else:
        angle = 0

    return center, radius, equation, point1, point2, distance, intersections_cv, angle

# def line_circle_intersection(xc, yc, r, x1, y1, x2, y2):
def line_circle_intersection(center_x, center_y, radius,x1, y1, x2, y2):
    # 计算直线斜率和截距
    if x2 - x1 == 0:
        slope = None
        intercept = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

    # 计算直线与圆的交点
    if slope is not None:
        A = 1 + slope ** 2
        B = -2 * center_x + 2 * slope * (intercept - center_y)
        C = center_x ** 2 + (intercept - center_y) ** 2 - radius ** 2

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return None
        else:
            x_1 = (-B + math.sqrt(discriminant)) / (2 * A)
            y_1 = slope * x_1 + intercept
            x_2 = (-B - math.sqrt(discriminant)) / (2 * A)
            y_2 = slope * x_2 + intercept

            return (x_1, y_1), (x_2, y_2)
    else:
        if abs(intercept - center_x) > radius:
            return None
        else:
            x_1 = intercept
            y_1 = center_y + math.sqrt(radius ** 2 - (intercept - center_x) ** 2)
            x_2 = intercept
            y_2 = center_y - math.sqrt(radius ** 2 - (intercept - center_x) ** 2)

            return (x_1, y_1), (x_2, y_2)

def rectangle_circumcircleV2(vertices,point1,point2):
    # 计算矩形的中心点
    center_x = (vertices[0][0] + vertices[1][0] + vertices[2][0] + vertices[3][0]) / 4
    center_y = (vertices[0][1] + vertices[1][1] + vertices[2][1] + vertices[3][1]) / 4
    center = (center_x, center_y)

    # 计算矩形对角线长度的一半，即为外接圆半径
    radius = math.sqrt((vertices[0][0] - center_x) ** 2 + (vertices[0][1] - center_y) ** 2)

    # 圆的方程
    equation = f'(x-{center_x})^2 + (y-{center_y})^2 = {radius**2}'


    # 计算两个采样点之间的距离
    point1, point2 = line_circle_intersection(center_x, center_y, radius, point1[0], point1[1], point2[0], point2[1])
    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # 计算两个采样点连成的直线的方程 y=k*x+b
    intersections = np.array(intersection(vertices, point1, point2))
    intersections = np.squeeze(intersections)
    angle = angle_with_x_axis(point1, point2)

    return center, radius, equation, point1, point2, distance, intersections, angle



def rotation_matrix(angle):
    # 将角度转换为弧度
    angle = math.radians(angle)
    # 计算旋转矩阵
    matrix = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
    # 返回结果
    return matrix

def inverse_matrix(matrix):
    # 计算逆矩阵
    inv_matrix = np.linalg.inv(matrix)
    # 返回结果
    return inv_matrix

def is_in_range(x,a,b):
    if a<=x<=b:
        return True
    else:
        return False