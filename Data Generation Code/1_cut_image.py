import os
import random

import cv2
import torch
import pickle
import open3d
import numpy as np
from scripts.circle import rectangle_circumcircle, is_in_range, check_list
import math
# 切割代码 

test_save_path = "/home/zrx/lab_disk1/zhourixin/oracle/make+fragment_zhouProcess/make_fragmentV4/test_image"
if os.path.exists(test_save_path) is False:
    os.mkdir(test_save_path)

class Fragment(object):
    def __init__(self, img, pcd, transformation, flag, area):
        self.img = img
        self.pcd = pcd
        self.trans = transformation
        self.flag = flag  # True 表示会继续再分，反之不分
        self.area = area


class Piecewise(object):
    @staticmethod
    def function(x, s):
        y = np.zeros(len(x))
        # x = np.roll(x, 1, axis=-1)


        if s[0] > s[1]:
            x = np.vstack((x[:, 1], x[:, 0])).transpose()
            width = s[1]
            height = s[0]
        else:
            width = s[0]
            height = s[1]
        
        # width = s[0]
        # height = s[1]    
        
        def f1():
            y=0
            for i in range(1, 20, 1):
                a = np.random.normal(height / 300, height / 150)
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )

            return y

        '''构造分段函数'''
        a = np.arange(int(width/6), int(width*5/6))
        b = np.arange(int(height*4/10), int(height*6/10))
        random_x = np.random.choice(a, 3, replace=False)
        random_x = np.sort(random_x)
        # 检查线段横坐标是否有小于width/8的，有的话就重新选择一次中间的衔接点
        while ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width/8).any():
            random_x = np.random.choice(a, 3, replace=False)
            random_x = np.sort(random_x)
        random_y = np.random.choice(b, 3)
        start_x, start_y = 0, np.random.randint(int(height*3/8), int(height*5/8))
        end_x, end_y = width, np.random.randint(int(height*3/8), int(height*5/8))
        
        p1 = (random_y[0] - start_y)/(random_x[0]-start_x) # p是斜率
        mask1 = x[:, 0] < random_x[0]
        y[mask1] += p1 * x[mask1][:, 0] - p1*random_x[0]+random_y[0]
        random_1=random.uniform(0,1)
        if random_1<1/2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
            y=y+f1()


        p2 = (random_y[1] - random_y[0])/(random_x[1]-random_x[0])
        mask2 = np.bitwise_and(random_x[0] <= x[:, 0], x[:, 0] < random_x[1])
        y[mask2] += p2 * x[mask2][:, 0] - p2*random_x[1]+random_y[1]
        random_2 = random.uniform(0, 1)
        if random_2 < 1 / 2:
            y = y + f1()

        p3 = (random_y[2] - random_y[1])/(random_x[2]-random_x[1])
        mask3 = np.bitwise_and(random_x[1] <= x[:, 0], x[:, 0] < random_x[2])
        y[mask3] += p3 * x[mask3][:, 0] - p3*random_x[2]+random_y[2]
        random_3 = random.uniform(0, 1)
        if random_3 < 1 / 2 :
            y = y + f1()

        p4 = (random_y[2] - end_y)/(random_x[2]-end_x)
        mask4 = x[:, 0] >= random_x[2]
        y[mask4] += p4 * x[mask4][:, 0] - p4*random_x[2]+random_y[2]
        random_4 = random.uniform(0, 1)
        if random_4 < 1 / 2 :
            y = y + f1()

        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)

        return mask_p_cover, mask_n_cover

def are_points_not_inside_contour(contour, points):
    points = np.array(points, dtype=np.int16)
    for point in points:
        dist = cv2.pointPolygonTest(contour, point, True)
        if dist <= 20:
            return True
    return False

def are_line_have_only_two_intertact(contour, line):
    line = np.array(line, dtype=np.int16)
    stat_list = []
    for L in line:
        dist = cv2.pointPolygonTest(contour, L, True)
        # if dist > 1:
        #     stat_list.append(1)
        # else:
        #     stat_list.append(-1)
        stat_list.append(dist)

        line_control_value = 30
        count = len([x for x in stat_list if 0 < x < line_control_value])
        if count > 3*line_control_value:
            return False
    return check_list(stat_list)


class Circle_cut_3point(object):
    @staticmethod
    def function(x, s,rotated_point,pcd_basic):
        y = np.zeros(len(x))
        paint_point = []
        paint_line = []
        # x = np.roll(x, 1, axis=-1)
        # if s[0] > s[1]:
        #     x = np.vstack((x[:, 1], x[:, 0])).transpose()
        #     width = s[1]
        #     height = s[0]
        # else:
        #     width = s[0]
        #     height = s[1]
        
        x = np.vstack((x[:, 1], x[:, 0])).transpose()
        width = s[1]
        height = s[0]

        # dist = cv2.pointPolygonTest(pcd_basic, (54,508), True)

        # width = s[0]
        # height = s[1]
        
        # def f1():
        #     y=0
        #     for i in range(1, 20, 1):
        #         a = np.random.normal(height / 300, height / 150)
        #         T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
        #         fi = np.random.uniform(-1, 1)
        #         w = 2 * np.pi / T
        #         y += \
        #             (
        #                     (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
        #             )

        #     return y
        def f1_point(point_x):
            y=0
            point_y = 0

            for i in range(1, 20, 1):
                a = np.random.normal(height / 150, height / 60) #300,150-》150 ,60
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )
                point_y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * point_x + fi)
                    )

            return y, point_y

        '''构造分段函数'''
        a = np.arange(int(width/6), int(width*5/6))
        # b = np.arange(int(height*4/10), int(height*6/10))
        b = np.arange(int(rotated_point[0][1]-int(height*1/20)), int(rotated_point[0][1]+int(height*1/20)))
        random_x = np.random.choice(a, 3, replace=False)
        random_x = np.sort(random_x)
        random_y = np.random.choice(b, 3)
        x_y_points = []
        for i in range(len(random_x)):
            x_y_points.append([random_x[i],random_y[i]])
        # if (random_x[0],random_y[0]) not in x:
        #     print(1)
        # while ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width/8 ).any() and \
        #       ( (random_x[0],random_y[0]) not in x) and ((random_x[1],random_y[1]) not in x) and ((random_x[2],random_y[2]) not in x):
        #     random_x = np.random.choice(a, 3, replace=False)
        #     random_x = np.sort(random_x)
        #     random_y = np.random.choice(b, 3)
        # 检查线段横坐标是否有小于width/8的，有的话就重新选择一次中间的衔接点
        error_count = 0
        while ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width/16).any() or \
            are_points_not_inside_contour(pcd_basic, x_y_points):
            error_count += 1
            if error_count > 10:
                return False
            random_x = np.random.choice(a, 3, replace=False)
            random_x = np.sort(random_x)
            random_y = np.random.choice(b, 3)
            x_y_points = []
            for i in range(len(random_x)):
                x_y_points.append([random_x[i],random_y[i]])
        # start_x, start_y = 0, np.random.randint(int(height*3/8), int(height*5/8))
        # end_x, end_y = width, np.random.randint(int(height*3/8), int(height*5/8))

        start_x, start_y = 0, int(rotated_point[0][1])
        end_x, end_y = width, int(rotated_point[0][1])


        ##################################   P1  #####################################
        p1 = (random_y[0] - start_y)/(random_x[0]-start_x) # p是斜率
        mask1 = x[:, 0] < random_x[0]
        y[mask1] += p1 * x[mask1][:, 0] - p1*random_x[0]+random_y[0]
        random_1=random.uniform(0,1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(start_x, random_x[0]):
            point_set.append([i, round(p1*(i-random_x[0])+random_y[0])])
        ##### 画出切割线 #####
        
        # if random_1<1/2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
        if random_1<1 / 2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
            point_x = np.arange(start_x, random_x[0])
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set
        ##### 画出切割线 #####
        ############################################################################



        ##################################   P2  #####################################
        p2 = (random_y[1] - random_y[0])/(random_x[1]-random_x[0])
        mask2 = np.bitwise_and(random_x[0] <= x[:, 0], x[:, 0] < random_x[1])
        y[mask2] += p2 * x[mask2][:, 0] - p2*random_x[1]+random_y[1]
        random_2 = random.uniform(0, 1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(random_x[0], random_x[1]):
            point_set.append([i, round(p2*(i-random_x[1])+random_y[1])])

        if random_2 < 1 / 2:
            point_x = np.arange(random_x[0], random_x[1])
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set





        ##################################   P3  #####################################
        p3 = (random_y[2] - random_y[1])/(random_x[2]-random_x[1])
        mask3 = np.bitwise_and(random_x[1] <= x[:, 0], x[:, 0] < random_x[2])
        y[mask3] += p3 * x[mask3][:, 0] - p3*random_x[2]+random_y[2]
        random_3 = random.uniform(0, 1)
        ##### 画出切割线 #####
        point_set = []
        for i in range(random_x[1], random_x[2]):
            point_set.append([i, round(p3*(i-random_x[2])+random_y[2])])
        
        
        if random_3 < 1 / 2:
            point_x = np.arange(random_x[1], random_x[2])
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set





        ##################################   P4  #####################################
        p4 = (random_y[2] - end_y)/(random_x[2]-end_x)
        mask4 = x[:, 0] >= random_x[2]
        y[mask4] += p4 * x[mask4][:, 0] - p4*random_x[2]+random_y[2]
        random_4 = random.uniform(0, 1)
        ##### 画出切割线 #####
        point_set = []
        for i in range(random_x[2], end_x):
            point_set.append([i, round(p4*(i-random_x[2])+random_y[2])])
        
        if random_4 < 1 / 2 :
            point_x = np.arange(random_x[2], end_x)
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set






        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)

        # 在旋转之后的图像上画出切割控制点
        paint_point.append((start_x, start_y))
        for i in range(len(random_x)):
            paint_point.append([random_x[i], random_y[i]])
        paint_point.append((end_x, end_y))

        # 判断切割线是否都在轮廓范围内
        if are_line_have_only_two_intertact(pcd_basic, paint_line) is False:
            return False

        # 如果长宽反转过，点的坐标也要反转
        # if s[0] <= s[1]:
        # paint_point_reverse, paint_line_reverse = [(y, x) for x, y in paint_point],[(y, x) for x, y in paint_line]
        # paint_point, paint_line = paint_point_reverse, paint_line_reverse
        return mask_p_cover, mask_n_cover, p_cover_area, n_cover_area, paint_point, paint_line

class Circle_cut_2point(object):
    @staticmethod
    def function(x, s,rotated_point,pcd_basic):
        y = np.zeros(len(x))
        paint_point = []
        paint_line = []
        # x = np.roll(x, 1, axis=-1)
        # if s[0] > s[1]:
        #     x = np.vstack((x[:, 1], x[:, 0])).transpose()
        #     width = s[1]
        #     height = s[0]
        # else:
        #     width = s[0]
        #     height = s[1]
        
        x = np.vstack((x[:, 1], x[:, 0])).transpose()
        width = s[1]
        height = s[0]

        # width = s[0]
        # height = s[1]
        
        def f1():
            y=0
            for i in range(1, 20, 1):
                a = np.random.normal(height / 300, height / 150)
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )

            return y
        def f1_point(point_x):
            y=0
            point_y = 0

            for i in range(1, 20, 1):
                a = np.random.normal(height / 150, height / 60) #300,150-》150 ,60 
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )
                point_y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * point_x + fi)
                    )

            return y, point_y

        '''构造分段函数'''
        a = np.arange(int(width/6), int(width*5/6))
        # b = np.arange(int(height*4/10), int(height*6/10))
        b = np.arange(int(rotated_point[0][1]-int(height*1/20)), int(rotated_point[0][1]+int(height*1/20)))
        random_x = np.random.choice(a, 2, replace=False)
        random_x = np.sort(random_x)
        random_y = np.random.choice(b, 2)
        x_y_points = []
        for i in range(len(random_x)):
            x_y_points.append([random_x[i],random_y[i]])

        error_count = 0
        while ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width/16).any() or \
            are_points_not_inside_contour(pcd_basic, x_y_points):
            error_count += 1
            if error_count > 10:
                return False
            random_x = np.random.choice(a, 2, replace=False)
            random_x = np.sort(random_x)
            random_y = np.random.choice(b, 2)
            x_y_points = []
            for i in range(len(random_x)):
                x_y_points.append([random_x[i],random_y[i]])
        # start_x, start_y = 0, np.random.randint(int(height*3/8), int(height*5/8))
        # end_x, end_y = width, np.random.randint(int(height*3/8), int(height*5/8))

        start_x, start_y = 0, int(rotated_point[0][1])
        end_x, end_y = width, int(rotated_point[0][1])


        ##################################   P1  #####################################
        p1 = (random_y[0] - start_y)/(random_x[0]-start_x) # p是斜率
        mask1 = x[:, 0] < random_x[0]
        y[mask1] += p1 * x[mask1][:, 0] - p1*random_x[0]+random_y[0]
        random_1=random.uniform(0,1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(start_x, random_x[0]):
            point_set.append([i, round(p1*(i-random_x[0])+random_y[0])])
        ##### 画出切割线 #####
        
        # if random_1<1/2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
        if random_1<1 / 2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
            point_x = np.arange(start_x, random_x[0])
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set
        ##### 画出切割线 #####
        ############################################################################



        ##################################   P2  #####################################
        p2 = (random_y[1] - random_y[0])/(random_x[1]-random_x[0])
        mask2 = np.bitwise_and(random_x[0] <= x[:, 0], x[:, 0] < random_x[1])
        y[mask2] += p2 * x[mask2][:, 0] - p2*random_x[1]+random_y[1]
        random_2 = random.uniform(0, 1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(random_x[0], random_x[1]):
            point_set.append([i, round(p2*(i-random_x[1])+random_y[1])])

        if random_2 < 1 / 2:
            point_x = np.arange(random_x[0], random_x[1])
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set




        ##################################   P3  #####################################
        p3 = (random_y[1] - end_y)/(random_x[1]-end_x)
        mask3 = random_x[1] <= x[:, 0]
        y[mask3] += p3 * x[mask3][:, 0] - p3*random_x[1]+random_y[1]
        random_3 = random.uniform(0, 1)
        ##### 画出切割线 #####
        point_set = []
        for i in range(random_x[1], end_x):
            point_set.append([i, round(p3*(i-random_x[1])+random_y[1])])
        
        
        if random_3 < 1 / 2:
            point_x = np.arange(random_x[1], end_x)
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set










        # ##################################   P4  #####################################
        # p4 = (random_y[2] - end_y)/(random_x[2]-end_x)
        # mask4 = x[:, 0] >= random_x[2]
        # y[mask4] += p4 * x[mask4][:, 0] - p4*random_x[2]+random_y[2]
        # random_4 = random.uniform(0, 1)
        # ##### 画出切割线 #####
        # point_set = []
        # for i in range(random_x[2], end_x):
        #     point_set.append([i, round(p4*(i-random_x[2])+random_y[2])])
        
        # if random_4 < 1 / 2 :
        #     point_x = np.arange(random_x[2], end_x)
        #     line_value, point_y = f1_point(point_x)
        #     y=y+line_value

        #     ##### 画出切割线 #####
        #     for i,item in enumerate(point_set):
        #         point_set[i][1] += round(point_y[i])
        # paint_line += point_set






        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)

        # 在旋转之后的图像上画出切割控制点
        paint_point.append((start_x, start_y))
        for i in range(len(random_x)):
            paint_point.append([random_x[i], random_y[i]])
        paint_point.append((end_x, end_y))

        # 判断切割线是否都在轮廓范围内
        if are_line_have_only_two_intertact(pcd_basic, paint_line) is False:
            return False

        

        # 如果长宽反转过，点的坐标也要反转
        # if s[0] <= s[1]:
        # paint_point_reverse, paint_line_reverse = [(y, x) for x, y in paint_point],[(y, x) for x, y in paint_line]
        # paint_point, paint_line = paint_point_reverse, paint_line_reverse
        return mask_p_cover, mask_n_cover, p_cover_area, n_cover_area, paint_point, paint_line

class Circle_cut_1point(object):
    @staticmethod
    def function(x, s,rotated_point,pcd_basic):
        y = np.zeros(len(x))
        paint_point = []
        paint_line = []
        # x = np.roll(x, 1, axis=-1)
        # if s[0] > s[1]:
        #     x = np.vstack((x[:, 1], x[:, 0])).transpose()
        #     width = s[1]
        #     height = s[0]
        # else:
        #     width = s[0]
        #     height = s[1]
        
        x = np.vstack((x[:, 1], x[:, 0])).transpose()
        width = s[1]
        height = s[0]

        # width = s[0]
        # height = s[1]
        
        def f1():
            y=0
            for i in range(1, 20, 1):
                a = np.random.normal(height / 300, height / 150)
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )

            return y
        def f1_point(point_x):
            y=0
            point_y = 0

            for i in range(1, 20, 1):
                a = np.random.normal(height / 150, height / 60) #300,150-》150 ,60
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )
                point_y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * point_x + fi)
                    )

            return y, point_y

        '''构造分段函数'''
        '''构造分段函数'''
        a = np.arange(int(width/6), int(width*5/6))
        # b = np.arange(int(height*4/10), int(height*6/10))
        b = np.arange(int(rotated_point[0][1]-int(height*1/20)), int(rotated_point[0][1]+int(height*1/20)))
        random_x = np.random.choice(a, 1, replace=False)
        random_x = np.sort(random_x)
        random_y = np.random.choice(b, 1)
        x_y_points = []
        for i in range(len(random_x)):
            x_y_points.append([random_x[i],random_y[i]])

        error_count = 0
        while ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width/16).any() or \
            are_points_not_inside_contour(pcd_basic, x_y_points):
            error_count += 1
            if error_count > 10:
                return False
            random_x = np.random.choice(a, 1, replace=False)
            random_x = np.sort(random_x)
            random_y = np.random.choice(b, 1)
            x_y_points = []
            for i in range(len(random_x)):
                x_y_points.append([random_x[i],random_y[i]])
        # start_x, start_y = 0, np.random.randint(int(height*3/8), int(height*5/8))
        # end_x, end_y = width, np.random.randint(int(height*3/8), int(height*5/8))

        start_x, start_y = 0, int(rotated_point[0][1])
        end_x, end_y = width, int(rotated_point[0][1])


        ##################################   P1  #####################################
        p1 = (random_y[0] - start_y)/(random_x[0]-start_x) # p是斜率
        mask1 = x[:, 0] < random_x[0]
        y[mask1] += p1 * x[mask1][:, 0] - p1*random_x[0]+random_y[0]
        random_1=random.uniform(0,1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(start_x, random_x[0]):
            point_set.append([i, round(p1*(i-random_x[0])+random_y[0])])
        ##### 画出切割线 #####
        
        # if random_1<1/2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
        if random_1<1 / 2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
            point_x = np.arange(start_x, random_x[0])
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set
        ##### 画出切割线 #####
        ############################################################################



        ##################################   P2  #####################################
        p2 = (random_y[0] - end_y)/(random_x[0]-end_x)
        mask2 = random_x[0] <= x[:, 0]
        y[mask2] += p2 * x[mask2][:, 0] - p2*random_x[0]+random_y[0]
        random_2 = random.uniform(0, 1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(random_x[0], end_x):
            point_set.append([i, round(p2*(i-random_x[0])+random_y[0])])

        if random_2 < 1 / 2:
            point_x = np.arange(random_x[0], end_x)
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set



        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)

        # 在旋转之后的图像上画出切割控制点
        paint_point.append((start_x, start_y))
        for i in range(len(random_x)):
            paint_point.append([random_x[i], random_y[i]])
        paint_point.append((end_x, end_y))

        # 判断切割线是否都在轮廓范围内
        if are_line_have_only_two_intertact(pcd_basic, paint_line) is False:
            return False

        # 如果长宽反转过，点的坐标也要反转
        # if s[0] <= s[1]:
        # paint_point_reverse, paint_line_reverse = [(y, x) for x, y in paint_point],[(y, x) for x, y in paint_line]
        # paint_point, paint_line = paint_point_reverse, paint_line_reverse
        return mask_p_cover, mask_n_cover, p_cover_area, n_cover_area, paint_point, paint_line

class Circle_cut_0point(object):
    @staticmethod
    def function(x, s,rotated_point,pcd_basic):
        y = np.zeros(len(x))
        paint_point = []
        paint_line = []
        # x = np.roll(x, 1, axis=-1)
        # if s[0] > s[1]:
        #     x = np.vstack((x[:, 1], x[:, 0])).transpose()
        #     width = s[1]
        #     height = s[0]
        # else:
        #     width = s[0]
        #     height = s[1]
        
        x = np.vstack((x[:, 1], x[:, 0])).transpose()
        width = s[1]
        height = s[0]

        # width = s[0]
        # height = s[1]
        
        def f1():
            y=0
            for i in range(1, 20, 1):
                a = np.random.normal(height / 300, height / 150)
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )

            return y
        def f1_point(point_x):
            y=0
            point_y = 0

            for i in range(1, 20, 1):
                a = np.random.normal(height / 150, height / 60) #300,150-》150 ,60
                T = np.random.normal(1.5 * width, 0.3 * width)  # T=(1/4)*width~6*width
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * x[:, 0] + fi)
                    )
                point_y += \
                    (
                            (a / (i + 1)) * np.sin(i * w * point_x + fi)
                    )

            return y, point_y

        '''构造分段函数'''
        # a = np.arange(int(width/6), int(width*5/6))
        # # b = np.arange(int(height*4/10), int(height*6/10))
        # b = np.arange(int(rotated_point[0][1]-rotated_point[0][1]*1/5), int(rotated_point[0][1]+rotated_point[0][1]*1/5))
        # # if len(b) == 0:
        # #     print
        # random_x = np.random.choice(a, 1, replace=False)
        # random_x = np.sort(random_x)
        # # 检查线段横坐标是否有小于width/8的，有的话就重新选择一次中间的衔接点
        # while ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width/8).any():
        #     random_x = np.random.choice(a, 1, replace=False)
        #     random_x = np.sort(random_x)
        # random_y = np.random.choice(b, 1)
        # start_x, start_y = 0, np.random.randint(int(height*3/8), int(height*5/8))
        # end_x, end_y = width, np.random.randint(int(height*3/8), int(height*5/8))

        start_x, start_y = 0, int(rotated_point[0][1])
        end_x, end_y = width, int(rotated_point[0][1])


        ##################################   P1  #####################################
        p1 = (end_y - start_y)/(end_x-start_x) # p是斜率
        # mask1 = x[:, 0] < random_x[0]
        y += p1 * x[:, 0] - p1*end_x+end_y
        random_1=random.uniform(0,1)

        ##### 画出切割线 #####
        point_set = []
        for i in range(start_x, end_x):
            point_set.append([i, round(p1*(i-end_x)+end_y)])
        ##### 画出切割线 #####
        
        # if random_1<1/2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
        if random_1<1 / 2 : # 先连接一条斜线，然后以0.5的概率决定要不要再往上叠加非线性函数
            point_x = np.arange(start_x, end_x)
            line_value, point_y = f1_point(point_x)
            y=y+line_value

            ##### 画出切割线 #####
            for i,item in enumerate(point_set):
                point_set[i][1] += round(point_y[i])
        paint_line += point_set
        ##### 画出切割线 #####
        ############################################################################



        ##################################   P2  #####################################
        # p2 = (random_y[0] - end_y)/(random_x[0]-end_x)
        # mask2 = random_x[0] <= x[:, 0]
        # y[mask2] += p2 * x[mask2][:, 0] - p2*random_x[0]+random_y[0]
        # random_2 = random.uniform(0, 1)

        # ##### 画出切割线 #####
        # point_set = []
        # for i in range(random_x[0], end_x):
        #     point_set.append([i, round(p2*(i-random_x[0])+random_y[0])])

        # if random_2 < 1 / 2:
        #     point_x = np.arange(random_x[0], end_x)
        #     line_value, point_y = f1_point(point_x)
        #     y=y+line_value

        #     ##### 画出切割线 #####
        #     for i,item in enumerate(point_set):
        #         point_set[i][1] += round(point_y[i])
        # paint_line += point_set



        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)

        # 在旋转之后的图像上画出切割控制点
        paint_point.append((start_x, start_y))
        # for i in range(len(random_x)):
        #     paint_point.append([random_x[i], random_y[i]])
        paint_point.append((end_x, end_y))

        # 判断切割线是否都在轮廓范围内
        if are_line_have_only_two_intertact(pcd_basic, paint_line) is False:
            return False

        # 如果长宽反转过，点的坐标也要反转
        # if s[0] <= s[1]:
        # paint_point_reverse, paint_line_reverse = [(y, x) for x, y in paint_point],[(y, x) for x, y in paint_line]
        # paint_point, paint_line = paint_point_reverse, paint_line_reverse
        return mask_p_cover, mask_n_cover, p_cover_area, n_cover_area, paint_point, paint_line

class Linear(object):
    @staticmethod
    def function(x, s):  # input: [前景点集，尺寸->用来翻转]
        y = np.zeros(len(x))
        '''根据尺寸翻转'''
        if s[0] > s[1]: #构造成横的是短边，竖的是长边
            x = np.vstack((x[:, 1], x[:, 0])).transpose()
            width = s[1]
            height = s[0]
        else:
            width = s[0]
            height = s[1]

        '''构造函数y = a*(x - 0.5*width) + 0.5*height'''
        a = np.random.randint(-50, 50) / 100  # 斜率[-0.5, 0.5]
        y += a * (x[:, 0] - 0.5*width) #0.5*width保证了是在中间位置开始分割的，但是斜率如何确定还有待商榷 
        y += height/2
        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)
        return mask_p_cover, mask_n_cover  # bool




class Multi_2(object):
    @staticmethod
    def function(x, s): # input: [前景点集，尺寸->用来翻转]
        y = np.zeros(len(x))
        if s[0] > s[1]:
            x = np.vstack((x[:, 1], x[:, 0])).transpose()
            width = s[1]
            height = s[0]
        else:
            width = s[0]
            height = s[1]
        for i in range(1, 20, 1):
            a = np.random.normal(height/40, height/150)
            T = np.random.normal(1.5*width, 0.3*width)   # T=(1/4)*width~6*width
            fi = np.random.uniform(-1, 1)
            w = 2 * np.pi / T

            # 存疑，为什么没有cos函数
            y += \
                (
                    (a / (i+1)) * np.sin(i * w * x[:, 0]+fi)
                )

        y += 0.5 * height
        res = y - x[:, 1]
        mask_p_cover = res <= 0
        mask_n_cover = res > 0

        p_cover_area = np.count_nonzero(mask_p_cover)
        n_cover_area = np.count_nonzero(mask_n_cover)

        return mask_p_cover, mask_n_cover

def save_fragment(frags, save_path, frag_idx, background):
    f_img = frags.img
    f_img[f_img[:, :, 3] == 0] = np.hstack((background, np.zeros(1, dtype=np.int)))
    f_pcd = frags.pcd
    f_trans = frags.trans
    cv2.imwrite(os.path.join(save_path, 'fragment {}.png'.format(str(idx).zfill(4))), f_img[:, :, :3])
    with open(os.path.join(save_path, 'gt.txt'), 'a') as f:
        f.write('{}\n'.format(frag_idx))
        f.write(str(f_trans.flatten())[1: -1].replace('\n', '') + '\n')
    # with open(os.path.join(save_path, 'pcd {}.pkl'.format(frag_idx)), 'wb') as f:
    #     pickle.dump(f_pcd, f)


def down_sample(pcd, stride):
    pcd = np.hstack((pcd, np.ones((len(pcd), 1))))
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(pcd)
    point_cloud = point_cloud.voxel_down_sample(stride)
    pcd = np.array(point_cloud.points)[:, 0:2]

    return pcd



## 转0度的版本
def rotate_func(angle, pcd, new, pad_=10):
    # angle = np.random.uniform(start, end)
    # angle = 0
    angle = math.radians(angle)
    cos_, sin_ = np.cos(angle), np.sin(angle)
    # x, y = (pcd[:, 0].max() + pcd[:, 0].min()) * 0.5, (pcd[:, 1].max() + pcd[:, 1].min()) * 0.5
    x, y = 0, 0
    # temp_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_],
    #                         [sin_, cos_, -x * sin_ - y * cos_]])
    temp_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_],
                            [sin_, cos_, -x * sin_ - y * cos_]])
    temp_pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), temp_matrix.T)
    shift_x = (0 - temp_pcd[:, 0].min())
    shift_y = (0 - temp_pcd[:, 1].min())
    # pcd = np.hstack((pcd[:, 1].reshape(-1, 1), pcd[:, 0].reshape(-1, 1)))
    rotate_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_ + shift_x + pad_],
                              [sin_, cos_, -x * sin_ - y * cos_ + shift_y + pad_]])

    pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), rotate_matrix.T)
    # pcd = np.hstack((pcd[:, 1].reshape(-1, 1), pcd[:, 0].reshape(-1, 1)))
    width_max, height_max = pcd[:, 0].max(), pcd[:, 1].max()
    
    
    cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(3).zfill(4))), new[:, :, :3])
    
    
    new = \
        cv2.warpAffine(new, rotate_matrix, (int(width_max) + pad_, int(height_max) + pad_), flags=cv2.INTER_NEAREST,
                       borderValue=0)

    cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(4).zfill(4))), new[:, :, :3])

    return pcd, new, rotate_matrix


# 旋转指定角度，旋转图像之后贴这边缩放
def image_rotate_funcV2(img, angle, intersections, pad_=10):
    height, width, channels = img.shape
    origin_angle = angle
    angle = np.radians(angle)
    
    basic_cover = (img[:, :, 3] != 0)
    basic_cover = (basic_cover == True).nonzero() #第一个数组包含所有非零元素的行索引，第二个数组包含所有非零元素的列索引
    basic_cover = np.vstack((basic_cover[0], basic_cover[1])).transpose()
    

    gray_p = np.zeros(img.shape[:2], dtype=np.uint8)
    gray_p[basic_cover[:, 0], basic_cover[:, 1]] = 255
    pcd_p, _ = cv2.findContours(gray_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # pcd_p, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if isinstance(pcd_p, tuple):
        len_list = list(map(len, pcd_p))
        min_len = min(len_list)
        if min_len >= 5 and len(len_list) >= 2:
            print('遇到问题，正在重试。。。')
            return False
        tar = pcd_p[0]
        for va in pcd_p:
            if len(va) > len(tar):
                tar = va
        pcd_p = tar
    pcd_p = np.asarray(pcd_p, dtype=np.float).reshape(-1, 2)
    pcd_p = down_sample(pcd_p, 10)
    pcd = pcd_p
    cos_, sin_ = np.cos(angle), np.sin(angle)
    # x, y = (pcd[:, 0].max() + pcd[:, 0].min()) * 0.5, (pcd[:, 1].max() + pcd[:, 1].min()) * 0.5
    x, y = 0, 0
    temp_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_],
                            [sin_, cos_, -x * sin_ - y * cos_]])
    temp_pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), temp_matrix.T)
    shift_x = (0 - temp_pcd[:, 0].min())
    shift_y = (0 - temp_pcd[:, 1].min())
    # pcd = np.hstack((pcd[:, 1].reshape(-1, 1), pcd[:, 0].reshape(-1, 1)))
    rotate_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_ + shift_x + pad_],
                              [sin_, cos_, -x * sin_ - y * cos_ + shift_y + pad_]])
    a = np.hstack((pcd, np.ones((len(pcd), 1))))
    pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), rotate_matrix.T)
    # pcd = np.hstack((pcd[:, 1].reshape(-1, 1), pcd[:, 0].reshape(-1, 1)))
    width_max, height_max = pcd[:, 0].max(), pcd[:, 1].max()

    img = \
        cv2.warpAffine(img, rotate_matrix, (int(width_max) + pad_, int(height_max) + pad_), flags=cv2.INTER_NEAREST,
                       borderValue=0)
    
    # print(intersections)
    a = np.hstack((intersections, np.ones((len(intersections), 1))))
    new_point = np.matmul(np.hstack((intersections, np.ones((len(intersections), 1)))), rotate_matrix.T)
    

    # return rotated_img, M
    rotate_matrix = np.vstack((rotate_matrix, (0,0,1)))
    return img, rotate_matrix, new_point


# 随机函数分割

def random_segmentation_circle(imgs):
    random_point=random.uniform(0,1)
    if is_in_range(random_point,0,8/16):
        f5 = Circle_cut_3point()  # 分段函数
    elif is_in_range(random_point,8/16,12/16):
        f5 = Circle_cut_2point()  # 分段函数
    elif is_in_range(random_point,12/16,14/16):
        f5 = Circle_cut_1point()  # 分段函数
    elif is_in_range(random_point,14/16,1):
        f5 = Circle_cut_0point()  # 分段函数
    # f5 = Circle_cut_3point()
    function = f5

    img = imgs
    if img.shape[-1] != 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    #在外接圆上随机采样两个点，然后连线，找连线和图像矩形的交点，然后得到连线与x轴正向的夹角，然后对图像进行旋转 
    height, width, channels = img.shape
    vertices = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
    while True:
        center, radius, equation, point1, point2, distance, intersections, angle = rectangle_circumcircle(vertices)
        # a = len(intersections)
        if distance > 0.9*2*radius and len(intersections)==2: # 采样点之间距离太近的话就重新采样
            
            break
    
    angle = 0 - angle

    # angle = 0
    
    cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(1).zfill(4))), img[:, :, :3])

    cur_return= image_rotate_funcV2(img, angle, intersections)
    if not cur_return:
        return False
    img, M, rotated_point = cur_return
    a = img.shape[0]
    if rotated_point[0][1] < 100 or rotated_point[0][1] > img.shape[0]-100: # 避免切割线太靠近边缘
        print('切割线太靠近边缘，正在重试。。。')
        return False
    roated_img = img
    
    cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(2).zfill(4))), img[:, :, :3])


    #计算背景颜色，每张图像的背景颜色都不相同（无实质作用，为了好看）
    bg = 255 - img[img[:, :, 3] != 0][:, :3].mean(0)
    bg = bg.astype(np.uint8)
    basic_cover = (img[:, :, 3] != 0)
    sizes = basic_cover.shape
    basic_cover = (basic_cover == True).nonzero() #第一个数组包含所有非零元素的行索引，第二个数组包含所有非零元素的列索引
    basic_cover = np.vstack((basic_cover[0], basic_cover[1])).transpose()

    gray_basic = np.zeros(img.shape[:2], dtype=np.uint8)
    gray_basic[basic_cover[:, 0], basic_cover[:, 1]] = 255
    pcd_basic, _ = cv2.findContours(gray_basic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if isinstance(pcd_basic, tuple):
        len_list = list(map(len, pcd_basic))
        if len_list:
            min_len = min(len_list)
            
            if min_len >= 5 and len(len_list) >= 2:
                print('遇到问题，正在重试。。。')
                return False
            # if len(len_list) >= 2:
            #     print('遇到问题，正在重试。。。')
            #     return False
            tar = pcd_basic[0]
            for va in pcd_basic:
                if len(va) > len(tar):
                    tar = va
            pcd_basic = tar
        else: 
            print('遇到问题，正在重试。。。')
            return False

    cut_result = function.function(basic_cover, sizes,rotated_point,pcd_basic)
    if cut_result is False:
        print('找不到轮廓内切割点，正在重试。。。')
        return False
    mask_p_cover, mask_n_cover, p_cover_area, n_cover_area, paint_point, paint_line = cut_result
    cover_pcd_p, cover_pcd_n = basic_cover[mask_p_cover], basic_cover[mask_n_cover]
    
    new_p = np.zeros(img.shape)
    gray_p = np.zeros(img.shape[:2], dtype=np.uint8)
    gray_p[cover_pcd_p[:, 0], cover_pcd_p[:, 1]] = 255
    new_p[cover_pcd_p[:, 0], cover_pcd_p[:, 1]] = img[cover_pcd_p[:, 0], cover_pcd_p[:, 1]]

    new_n = np.zeros(img.shape)
    gray_n = np.zeros(img.shape[:2], dtype=np.uint8)
    gray_n[cover_pcd_n[:, 0], cover_pcd_n[:, 1]] = 255
    new_n[cover_pcd_n[:, 0], cover_pcd_n[:, 1]] = img[cover_pcd_n[:, 0], cover_pcd_n[:, 1]]
    
    new_p, new_n = new_p.astype(np.uint8), new_n.astype(np.uint8)

    pcd_p, _ = cv2.findContours(gray_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pcd_n, _ = cv2.findContours(gray_n, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if isinstance(pcd_p, tuple):
        len_list = list(map(len, pcd_p))

        if len_list:
            min_len = min(len_list)
            if min_len >= 2 and len(len_list) >= 2:
                print('遇到问题，正在重试。。。')
                return False
            # if len(len_list) >= 2:
            #     print('遇到问题，正在重试。。。')
            #     return False
            tar = pcd_p[0]
            for va in pcd_p:
                if len(va) > len(tar):
                    tar = va
            pcd_p = tar
        else:
            print('遇到问题，正在重试。。。')
            return False

    if isinstance(pcd_n, tuple):
        len_list = list(map(len, pcd_n))
        if len_list:
            min_len = min(len_list)
            
            if min_len >= 5 and len(len_list) >= 2:
                print('遇到问题，正在重试。。。')
                return False
            # if len(len_list) >= 2:
            #     print('遇到问题，正在重试。。。')
            #     return False
            tar = pcd_n[0]
            for va in pcd_n:
                if len(va) > len(tar):
                    tar = va
            pcd_n = tar
        else: 
            print('遇到问题，正在重试。。。')
            return False
    try:
        pcd_p = np.asarray(pcd_p, dtype=np.float).reshape(-1, 2)
        pcd_n = np.asarray(pcd_n, dtype=np.float).reshape(-1, 2)
        pcd_p, pcd_n = down_sample(pcd_p, 10), down_sample(pcd_n, 10)  # 下采样
        # pcd_p, pcd_n = ordering_point(pcd_p), ordering_point(pcd_n)  # 获取有顺序的排序
        # pcd_p, pcd_n = contour_interpolation(pcd_p, 8), contour_interpolation(pcd_n, 8)  # 轮廓平滑插值
    except:
        print('遇到问题，正在重试。。。')
        return False
    '''随机旋转图片并获得对应点集图片和旋转矩阵'''
    ## 这一步没有对碎片进行缩放，但是将外包围从原来的整幅图像调整为包围住轮廓，再往外扩展一定像素的框
    # pcd_n, new_n, rotate_matrix_n = rotate_func(0, pcd_n, new_n, pad_=10)
    # pcd_p, new_p, rotate_matrix_p = rotate_func(0, pcd_p, new_p, pad_=10)
    pcd_n, new_n, rotate_matrix_n = rotate_func(0-angle, pcd_n, new_n, pad_=10)
    pcd_p, new_p, rotate_matrix_p = rotate_func(0-angle, pcd_p, new_p, pad_=10)
    cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(5).zfill(4))), new_n[:, :, :3])
    cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(6).zfill(4))), new_p[:, :, :3])
    rotate_matrix_p = np.vstack((rotate_matrix_p, np.array([0, 0, 1])))
    rotate_matrix_n = np.vstack((rotate_matrix_n, np.array([0, 0, 1])))

    rotate_matrix_p_nn = np.matmul(rotate_matrix_p, M)
    rotate_matrix_n_nn = np.matmul(rotate_matrix_n, M)

    # rotate_matrix_p[0][0],rotate_matrix_p[0][1],rotate_matrix_p[0][2],rotate_matrix_p[1][0],rotate_matrix_p[1][1],rotate_matrix_p[1][2] = 1,0,rotate_matrix_p[0][2]+M[0][2],0,1,rotate_matrix_p[1][2]+M[1][2]
    # rotate_matrix_n[0][0],rotate_matrix_n[0][1],rotate_matrix_n[0][2],rotate_matrix_n[1][0],rotate_matrix_n[1][1],rotate_matrix_n[1][2] = 1,0,rotate_matrix_n[0][2]+M[0][2],0,1,rotate_matrix_n[1][2]+M[1][2]

    # rotate_matrix_p = np.linalg.inv(np.vstack((rotate_matrix_p, np.array([0, 0, 1]))))
    # rotate_matrix = np.matmul(np.vstack((rotate_matrix_n, np.array([0, 0, 1]))), rotate_matrix_p)[:2]
    # 可视化数据，测试获取的数据集是否有问题
    # pcd_p, pcd_n = pcd_p.astype(np.int), pcd_n.astype(np.int)
    # new_p[pcd_p[:, 0], pcd_p[:, 1]] = 0
    # new_n[pcd_n[:, 0], pcd_n[:, 1]] = 0
    # cv2.imwrite('./made_datav2/{}a.png'.format(i), new_p)
    # cv2.imwrite('./made_datav2/{}b.png'.format(i), new_n)
    '''调整轮廓为逆时针方向'''
    pcd_p_rstep = np.roll(pcd_p, 1, axis=0)
    x_mean_, y_mean_ = pcd_p[:, 0].mean(), pcd_p[:, 1].mean()
    sample_vec = pcd_p - pcd_p_rstep
    normal = pcd_p - np.array([x_mean_, y_mean_])
    if np.cross(sample_vec, normal).mean() > 0:
        pcd_p = pcd_p[::-1]

    pcd_n_rstep = np.roll(pcd_n, 1, axis=0)
    x_mean_, y_mean_ = pcd_n[:, 0].mean(), pcd_n[:, 1].mean()
    sample_vec = pcd_n - pcd_n_rstep
    normal = pcd_n - np.array([x_mean_, y_mean_])
    if np.cross(sample_vec, normal).mean() > 0:
        pcd_n = pcd_n[::-1]
    
    if paint_point:
        # 在图像上画出红色的坐标点
        for point in paint_point:
            cv2.circle(roated_img, point, radius=5, color=(0, 0, 255), thickness=-1)
    if paint_line:
        # 在图像上画出白色的切割线
        for point in paint_line:
            cv2.circle(roated_img, point, radius=1, color=(255, 255, 255), thickness=-1)

    return new_p, new_n, pcd_p, pcd_n, rotate_matrix_p_nn, rotate_matrix_n_nn, bg, p_cover_area, n_cover_area, roated_img  # new -> 分割得到的图片, pcd -> 轮廓点, p和n分别是被分开的两个碎片



# f1 = Multi_1()  # 平缓的不规则函数


# f2 = Piecewise_circle()  # 分段函数
# f3 = Linear()  # 线性函数
# f4 = Multi_2()
# random_point=random.uniform(0,1)
# if is_in_range(random_point,0,8/16):
#     f5 = Circle_cut_3point()  # 分段函数
# elif is_in_range(random_point,8/16,12/16):
#     f5 = Circle_cut_2point()  # 分段函数
# elif is_in_range(random_point,12/16,14/16):
#     f5 = Circle_cut_1point()  # 分段函数
# elif is_in_range(random_point,14/16,1):
#     f5 = Circle_cut_0point()  # 分段函数
# func_list = [f5] 
# f5 = Circle_cut_3point()
# f5 = Circle_cut_2point()
# f5 = Circle_cut_1point()
# f5 = Circle_cut_0point()

 # 储存所有需要用到的函数对象
# func_list = [f2]
'------------------------------main_func----------------------------------------'
if __name__ == '__main__':
    print('正在处理数据目录')

    root = '/home/zrx/lab_disk1/zhourixin/oracle/DATASET/image+all+in+one/image all in one'  # 所有包含一张图片的文件夹的路径
    # root = '/home/zrx/lab_disk1/zhourixin/oracle/make+fragment/make fragment/my dataset/all/car'

    # segment_logic = "origin"
    # segment_logic = "circle_sample_V5"
    # segment_logic = "circle_sample_V5_1"# V5基础上，增大幅度 /150->/80
    segment_logic = "circle_sample_V5_2"# V5_1基础上，限制切割点必须在轮廓范围内
    print(f"cut mode:{segment_logic}")




    if segment_logic == "circle_sample_V5_2":
        save_root = "/home/zrx/lab_disk1/zhourixin/oracle/DATASET/image+all+in+one/image all in one "+segment_logic + "/fragments"
        process_root = "/home/zrx/lab_disk1/zhourixin/oracle/DATASET/image+all+in+one/image all in one "+segment_logic + "/process"
        areaJPG_root = "/home/zrx/lab_disk1/zhourixin/oracle/DATASET/image+all+in+one/image all in one "+segment_logic
        img_list = os.listdir(root)
        area_list=np.array([])

        max_area = 0
        min_area = 1e8
        print("max_area:%f, min_area:%f" % (max_area, min_area))
        for m in range(len(img_list)):
            res_list = []  # 储存所有不再继续分割的碎片的对象的列表

            
            print('正在处理第 {} 张图片'.format(m))
            idx = 0
            '''读取原图'''
            img_path = os.path.join(root, img_list[m])
            img1 = cv2.imread(os.path.join(img_path, 'image.jpg'), cv2.IMREAD_UNCHANGED)

            this_image_area_list = []
            this_image_area = img1.size / 3

            saved_path = os.path.join(save_root, img_list[m])
            if os.path.exists(saved_path) is False:
                os.makedirs(saved_path)

            cv2.imwrite(os.path.join(saved_path, 'image.jpg'), img1)

            fragment_list = []

            # img_p, img_n, pcd_p, pcd_n, temp_trans_p, temp_trans_n, bg = random_segmentation_circle(img1, func_list)
            cur_res = random_segmentation_circle(img1)
            error_count = 0
            while not cur_res:
                cur_res = random_segmentation_circle(img1)
                error_count += 1
                if error_count > 10:
                    break
            if not cur_res:
                continue
            img_p, img_n, pcd_p, pcd_n, temp_trans_p, temp_trans_n, bg, p_cover_area, n_cover_area, roated_img = cur_res

            print('第 {} 块碎片初始分割完成'.format(m))
            fragment_p = Fragment(img_p, pcd_p, temp_trans_p, True, p_cover_area)
            fragment_n = Fragment(img_n, pcd_n, temp_trans_n, True, n_cover_area)
            fragment_list.append(fragment_p)
            fragment_list.append(fragment_n)

            # img_p_area=len(np.where(img_p[:,:,-3]!=0)[0])
            # img_n_area=len(np.where(img_n[:,:,-3]!=0)[0])
            this_image_area_list.append(p_cover_area/this_image_area)
            this_image_area_list.append(n_cover_area/this_image_area)


            for _ in range(35):
            # while any(x > 0.5 for x in this_image_area_list):

                cur_fragment = random.choices(fragment_list, weights=this_image_area_list, k=1)[0]
                cur_idx = fragment_list.index(cur_fragment)

                cur_img = cur_fragment.img
                cur_trans = cur_fragment.trans
                cur_res = random_segmentation_circle(cur_img)
                error_count = 0
                while not cur_res:
                    cur_res = random_segmentation_circle(cur_img)
                    error_count += 1
                    if error_count > 8:
                        break
                if not cur_res:
                    continue

                cur_img_p, cur_img_n, cur_pcd_p, cur_pcd_n, cur_trans_p, cur_trans_n, _, p_cover_area, n_cover_area, roated_img = cur_res
                cur_trans_p = np.matmul(cur_trans_p, cur_trans)
                cur_trans_n = np.matmul(cur_trans_n, cur_trans)

                # cur_img_p_area=len(np.where(cur_img_p[:,:,-3]!=0)[0])#大面积的黑色就有问题了；长和宽分别来卡
                # cur_img_n_area=len(np.where(cur_img_n[:,:,-3]!=0)[0])

                # if cur_img_p_area < 100**2 or cur_img_n_area < 150**2:
                #     continue
                Limit = 150
                # print(cur_img_p.shape[0],cur_img_p.shape[1],cur_img_n.shape[0],cur_img_n.shape[1])
                edge_scale_min_p = min(cur_img_p.shape[0]/cur_img_p.shape[1],cur_img_p.shape[1]/cur_img_p.shape[0])
                edge_scale_max_p = max(cur_img_p.shape[1]/cur_img_p.shape[0],cur_img_p.shape[0]/cur_img_p.shape[1])

                edge_scale_min_n = min(cur_img_n.shape[0]/cur_img_n.shape[1],cur_img_n.shape[1]/cur_img_n.shape[0])
                edge_scale_max_n = max(cur_img_n.shape[1]/cur_img_n.shape[0],cur_img_n.shape[0]/cur_img_n.shape[1])

                area_scale_p = p_cover_area/(cur_img_p.shape[0]*cur_img_p.shape[1])
                area_scale_n = n_cover_area/(cur_img_n.shape[0]*cur_img_n.shape[1])


                if cur_img_p.shape[0] < Limit or cur_img_p.shape[1] < Limit \
                    or cur_img_n.shape[0] < Limit or cur_img_n.shape[1] < Limit \
                    or p_cover_area < Limit**2 or n_cover_area < Limit**2 \
                    or edge_scale_max_p>4 or edge_scale_min_p<0.25\
                    or edge_scale_max_n>4 or edge_scale_min_n<0.25\
                    or area_scale_p<0.3 or area_scale_n<0.3:
                    continue
                    
                # 保存中间的切割过程
                processed_img = roated_img
                process_save_path = os.path.join(process_root, img_list[m])
                cut_save_path = os.path.join(process_save_path, "cut_"+str(idx).zfill(4))
                if os.path.exists(cut_save_path) is False:
                    os.makedirs(cut_save_path)
                cv2.imwrite(os.path.join(cut_save_path, 'origin.png'), processed_img[:, :, :3])
                cv2.imwrite(os.path.join(cut_save_path, 'cut1.png'), cur_img_p[:, :, :3])
                cv2.imwrite(os.path.join(cut_save_path, 'cut2.png'), cur_img_n[:, :, :3])

                '''Set segmentation possibility'''
                flag_p = True if np.random.randint(0, 100) < 40 else False
                flag_n = True if np.random.randint(0, 100) < 40 else False
                fragment_p = Fragment(cur_img_p, cur_pcd_p, cur_trans_p, flag_p, p_cover_area)
                fragment_n = Fragment(cur_img_n, cur_pcd_n, cur_trans_n, flag_n, n_cover_area)
                fragment_list.append(fragment_p)
                fragment_list.append(fragment_n)


                # img_p_area=len(np.where(cur_img_p[:,:,-3]!=0)[0])
                # img_n_area=len(np.where(cur_img_n[:,:,-3]!=0)[0])
                this_image_area_list.append(p_cover_area/this_image_area)
                this_image_area_list.append(n_cover_area/this_image_area)

                if p_cover_area > max_area:
                    max_area = p_cover_area
                if p_cover_area < min_area:
                    min_area = p_cover_area
                if n_cover_area > max_area:
                    max_area = n_cover_area
                if n_cover_area < min_area:
                    min_area = n_cover_area

                idx += 1
                del fragment_list[cur_idx]
                del this_image_area_list[cur_idx]
                print('已拆分第 {} 张图 的第 {} 块碎片'.format(m, idx+1))
                

            # total = 0
            # for i in this_image_area_list:
            #     total += i
            #如果有大碎片的话再挑出来切割一下
            area_thresh = 0.15
            while any(x > area_thresh for x in this_image_area_list):
                result = [i for i, x in enumerate(this_image_area_list) if x > area_thresh]
                big_idx = result[0]
                
                for _ in range(15):
                    cur_idx = big_idx
                    cur_fragment = fragment_list[cur_idx]
                    cur_img = cur_fragment.img
                    cur_trans = cur_fragment.trans
                    cur_res = random_segmentation_circle(cur_img)
                    error_count = 0
                    while not cur_res:
                        cur_res = random_segmentation_circle(cur_img)
                        error_count += 1
                        if error_count > 10:
                            break
                    if not cur_res:
                        break

                    cur_img_p, cur_img_n, cur_pcd_p, cur_pcd_n, cur_trans_p, cur_trans_n, _, p_cover_area, n_cover_area, roated_img = cur_res
                    cur_trans_p = np.matmul(cur_trans_p, cur_trans)
                    cur_trans_n = np.matmul(cur_trans_n, cur_trans)

                    # cur_img_p_area=len(np.where(cur_img_p[:,:,-3]!=0)[0])#大面积的黑色就有问题了；长和宽分别来卡
                    # cur_img_n_area=len(np.where(cur_img_n[:,:,-3]!=0)[0])

                    # if cur_img_p_area < 100**2 or cur_img_n_area < 150**2:
                    #     continue
                    Limit = 150
                    # print(cur_img_p.shape[0],cur_img_p.shape[1],cur_img_n.shape[0],cur_img_n.shape[1])
                    edge_scale_min_p = min(cur_img_p.shape[0]/cur_img_p.shape[1],cur_img_p.shape[1]/cur_img_p.shape[0])
                    edge_scale_max_p = max(cur_img_p.shape[1]/cur_img_p.shape[0],cur_img_p.shape[0]/cur_img_p.shape[1])

                    edge_scale_min_n = min(cur_img_n.shape[0]/cur_img_n.shape[1],cur_img_n.shape[1]/cur_img_n.shape[0])
                    edge_scale_max_n = max(cur_img_n.shape[1]/cur_img_n.shape[0],cur_img_n.shape[0]/cur_img_n.shape[1])

                    area_scale_p = p_cover_area/(cur_img_p.shape[0]*cur_img_p.shape[1])
                    area_scale_n = n_cover_area/(cur_img_n.shape[0]*cur_img_n.shape[1])


                    if cur_img_p.shape[0] < Limit or cur_img_p.shape[1] < Limit \
                        or cur_img_n.shape[0] < Limit or cur_img_n.shape[1] < Limit \
                        or p_cover_area < Limit**2 or n_cover_area < Limit**2 \
                        or edge_scale_max_p>4 or edge_scale_min_p<0.25\
                        or edge_scale_max_n>4 or edge_scale_min_n<0.25\
                        or area_scale_p<0.3 or area_scale_n<0.3:
                        continue
                        
                    # 保存中间的切割过程
                    processed_img = roated_img
                    process_save_path = os.path.join(process_root, img_list[m])
                    cut_save_path = os.path.join(process_save_path, "cut_"+str(idx).zfill(4))
                    if os.path.exists(cut_save_path) is False:
                        os.makedirs(cut_save_path)
                    cv2.imwrite(os.path.join(cut_save_path, 'origin.png'), processed_img[:, :, :3])
                    cv2.imwrite(os.path.join(cut_save_path, 'cut1.png'), cur_img_p[:, :, :3])
                    cv2.imwrite(os.path.join(cut_save_path, 'cut2.png'), cur_img_n[:, :, :3])

                    '''Set segmentation possibility'''
                    flag_p = True if np.random.randint(0, 100) < 40 else False
                    flag_n = True if np.random.randint(0, 100) < 40 else False
                    fragment_p = Fragment(cur_img_p, cur_pcd_p, cur_trans_p, flag_p, p_cover_area)
                    fragment_n = Fragment(cur_img_n, cur_pcd_n, cur_trans_n, flag_n, n_cover_area)
                    fragment_list.append(fragment_p)
                    fragment_list.append(fragment_n)


                    # img_p_area=len(np.where(cur_img_p[:,:,-3]!=0)[0])
                    # img_n_area=len(np.where(cur_img_n[:,:,-3]!=0)[0])
                    this_image_area_list.append(p_cover_area/this_image_area)
                    this_image_area_list.append(n_cover_area/this_image_area)

                    if p_cover_area > max_area:
                        max_area = p_cover_area
                    if p_cover_area < min_area:
                        min_area = p_cover_area
                    if n_cover_area > max_area:
                        max_area = n_cover_area
                    if n_cover_area < min_area:
                        min_area = n_cover_area

                    idx += 1
                    del fragment_list[cur_idx]
                    del this_image_area_list[cur_idx]
                    print('已拆分第 {} 张图 的第 {} 块大碎片'.format(m, big_idx))
                    break
                if error_count > 10:
                    this_image_area_list[big_idx] = 0 # 反复拆分都失败的大碎片，就不切了
                # if cut_big_frag_idx > 14 or error_count > 10:
                #     print("第 big_idx 张大碎片切割失败")

                # print(result)

            print('分割完毕，正在处理中')
            # for frag in fragment_list:
            #     pcd, img, trans = rotate_func(-90, 90,  frag.pcd, frag.img, pad_=20)
            #     trans = np.vstack((trans, np.array([0, 0, 1])))
            #     frag.img = img
            #     frag.pcd = pcd
            #     frag.trans = np.matmul(trans, frag.trans)

            for frag in fragment_list:
                area=frag.area
                area_list=np.append(area_list,area)
            print('保存中。。。')

            for idx in range(len(fragment_list)):
                save_fragment(fragment_list[idx], saved_path, idx, bg)
            with open(os.path.join(saved_path, 'bg.txt'), 'w') as f:
                f.write(str(bg)[1:-1])

        print('全部处理完毕，程序结束')
        print("max_area:%f, min_area:%f" % (max_area, min_area))
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.displot(area_list)

        plt.savefig(areaJPG_root+'/area.jpg')
        plt.show()

        with open(os.path.join(areaJPG_root, 'area_range.txt'), 'w') as f:
            f.write(("max_area:%f, min_area:%f" % (max_area, min_area)))


