import os
import cv2
import pickle
import string
import open3d
import numpy as np
from math import fabs
from scripts import data_preprocess
from PIL import Image
count1 = 0

# 埃尔米特插值
"""
Hermite 插值公式是一种多项式插值方法，它推广了拉格朗日插值。拉格朗日插值允许计算一个小于 n 次的多项式，
使其在 n 个给定点处的值与给定函数相同。相反, Hermite 插值计算一个小于 mn 次的多项式，
使该多项式及其前 m-1 阶导数在 n 个给定点处的值与给定函数及其前 m-1 阶导数相同
Hermite 插值多项式的 x 坐标和 y 坐标分别为：
x(t) = (2t^3-3t^2+1)*p0[0] + (-2t^3+3t^2)*p1[0] + (t^3-2t^2+t)*r0[0] + (t^3-t^2)*r1[0]
y(t) = (2t^3-3t^2+1)*p0[1] + (-2t^3+3t^2)*p1[1] + (t^3-2t^2+t)*r0[1] + (t^3-t^2)*r1[1]
"""
def hermite(p0, p1, r0, r1):
    """
    to interpolated points between two points by  using hermite interpolation.
    :param p0:
    :param p1:
    :param r0:
    :param r1:
    :return: interpolated point between two points.
    """
    distance = np.linalg.norm(p1 - p0)
    stride = 1 / distance
    t = np.arange(0, 1, stride)
    T = np.array([t ** 3, t ** 2, t ** 1, np.ones(len(t))]).T
    M = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]]) # 插值多项式的系数
    G = np.array([p0, p1, r0, r1]) # 几何矩阵
    Fh = np.matmul(T, M)
    new = np.matmul(Fh, G)

    return new

def draw_points(points, image):
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    for point in points:
        x, y = point
        blank_image[y, x] = [255, 255, 255]


    return np.array(blank_image)

# 这段代码的目的是对给定的有序点集进行插值和下采样，并返回插值和下采样后的点集。
def contour_interpolation(order_point, stride):
    """
    to interpolate an ordered point set.
    :param order_point: an ordered point set. type = Ndarray.
    :param stride: sampling interval of point set
    :return: an interpolated point set. type = Ndarray.
    """
    new_point = np.zeros((0, 2))
    order_point = order_point[::10]
    for m in range(len(order_point)):
        if m >= len(order_point):
            break
        if np.linalg.norm(order_point[0] - order_point[-1]) < 0.5 * 10:
            order_point = order_point[1:]
        point = [order_point[m + n - 3] for n in range(4)] # 每次找四个点出来
        point = np.array(point)
        ii = 1
        jj = 2
        p0 = point[ii] # 拿出四个点中间的两个点
        p1 = point[jj]
        r0 = (point[ii + 1] - point[ii - 1]) / 2
        r1 = (point[jj + 1] - point[jj - 1]) / 2
        points = hermite(p0, p1, r0, r1)
        new_point = np.vstack((new_point, points))

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.hstack((new_point, np.ones((len(new_point), 1)))))
    full_new_point = np.array(point_cloud.points)[:, :2]
    downsample = point_cloud.uniform_down_sample(stride) #均匀采样
    new_point = np.array(downsample.points)[:, :2]

    return new_point, full_new_point, downsample


def file_filter(f):
    if f[-4:] in ['.png']:
        return True
    else:
        return False

test_image_path= "./"
def save_test_image(saved_image, origin_image, image_id, forder_name):
    if forder_name in ["origin_b_image", "dilated_image"] :
        img = Image.fromarray(saved_image.astype('uint8'))
        img.save(test_image_path+'/{}/contour_order_{}.png'.format(forder_name, image_id))
    else:
        draw_countour = saved_image.astype('int32')
        draw_countour = draw_points(draw_countour, origin_image)
        img = Image.fromarray(draw_countour.astype('uint8'))
        img.save(test_image_path+'/{}/contour_order_{}.png'.format(forder_name, image_id))

def down_sample(order_point, stride):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.hstack((order_point, np.ones((len(order_point), 1)))))
    downsample = point_cloud.uniform_down_sample(stride) #均匀采样
    new_point = np.array(downsample.points)[:, :2]

    return new_point

def preprocess(path):
    global count1
    imglist = os.listdir(path)
    imglist = list(filter(file_filter, imglist))
    imglist.sort(key=lambda x: int(x[:-4][9:]))
    transforms = np.zeros((0, 9))
    with open(os.path.join(path, 'gt.txt'), 'r') as gt_file:
        while True:
            transform = gt_file.readline()
            if not transform:
                break
            else:
                transform = string.capwords(transform.strip()).split(' ')
                if len(transform) == 1:
                    continue
                else:
                    transform = np.asarray(transform, dtype=np.float)
                    transforms = np.vstack((transforms, transform[:9]))

    transforms = transforms.reshape(-1, 3, 3)
    transforms = np.linalg.inv(transforms)

    '''get org images'''
    img_all = []
    extra_all = []
    shapes = np.zeros((0, 3), dtype=np.int)
    for i in range(len(imglist)):
        if imglist[i][-3:] != 'png':
            continue
        if imglist[i][:8] != 'fragment':
            continue
        img = cv2.imread(os.path.join(img_path, imglist[i]), cv2.IMREAD_UNCHANGED)
        img = img.transpose(1, 0, 2)
        img_all.append(img)
        shapes = np.vstack((shapes, img.shape))
        with open(os.path.join(img_path, 'bg.txt'), 'r') as bg_f:
            bg = bg_f.readline()
        bg = np.asarray(bg.split(), dtype=np.int)

    '''get image contour'''
    full_contour_all = []
    down_sample_contour = []
    for j in range(len(img_all)):
        
        print('fragment {} start'.format(j+1))
        image = img_all[j]

        image_save = Image.fromarray(image.astype('uint8') )
        image_save.save(test_image_path+'/origin0/image_{}.png'.format(j))

        mask = (image == bg).all(axis=-1)
        image[mask] = (0, 0, 0)  # 图片背景设置为0
        img_all[j] = image.transpose(1, 0, 2)
        gray = np.ones(image.shape[:2], dtype=np.uint8)
        gray[~mask] = 255
        _, b_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # save_test_image(b_image, image, j, "origin_b_image")
        contour, hierarchy = cv2.findContours(b_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contour) > 1:
            # temp_contour = np.zeros((0, 2))
            # for c in contour:
            #     temp_contour = np.vstack((temp_contour, c.reshape(-1, 2)))
            # contour = temp_contour
            max_len_contour=-1
            max_contour=contour[0]
            count1 += 1
            for i,c in enumerate(contour):
                if len(c)>=max_len_contour:
                    max_contour=contour[i]
                    max_len_contour=len(c)
            contour=max_contour.reshape(-1, 2)



        else:
            contour = np.asarray(contour, dtype=np.float).reshape(-1, 2)
        contour_order = contour.copy()

        stride = 3
        sigma = 3
        temp_contour = contour_order.copy()
        temp_contour = np.expand_dims(temp_contour, axis=1)
        contour_guss = cv2.GaussianBlur(temp_contour.astype(np.float32), (0, 0), sigma)
        contour_guss = np.squeeze(contour_guss)
        contour_guss = contour_guss.astype(int)
        mask = np.linalg.norm(contour_guss - np.roll(contour_guss, 1, axis=0), axis=-1) == 0
        contour_guss = contour_guss[~mask]
        down_sample_guss_point = down_sample(contour_guss, stride)

        '''counterclockwise downsample contour'''
        contour_order_rstep = np.roll(down_sample_guss_point, 1, axis=0)
        x_mean_, y_mean_ = down_sample_guss_point[:, 0].mean(), down_sample_guss_point[:, 1].mean()
        sample_vec = down_sample_guss_point - contour_order_rstep
        normal = down_sample_guss_point - np.array([x_mean_, y_mean_])
        if np.cross(sample_vec, normal).mean() > 0:
            down_sample_guss_point = down_sample_guss_point[::-1]
        else:
            pass
        '''counterclockwise guss contour'''
        contour_order_rstep = np.roll(contour_guss, 1, axis=0)
        x_mean_, y_mean_ = contour_guss[:, 0].mean(), contour_guss[:, 1].mean()
        sample_vec = contour_guss - contour_order_rstep
        normal = contour_guss - np.array([x_mean_, y_mean_])
        if np.cross(sample_vec, normal).mean() > 0:
            contour_guss = contour_guss[::-1]
        else:
            pass

        full_contour_all.append(contour_guss)
        down_sample_contour.append(down_sample_guss_point)
    print(count1)

    matching_set['full_pcd_all'].extend(full_contour_all)
    matching_set['img_all'].extend(img_all)
    matching_set['extra_img'].extend(extra_all)
    matching_set['shape_all'].extend(list(shapes))
    matching_set["down_sample_pcd"].append(down_sample_contour)

    for i in range(len(img_all)):
        for k in range(i+1, len(img_all), 1):
            t1, t2 = transforms[i][:2], transforms[k][:2]
            contour1, contour2 = full_contour_all[i], full_contour_all[k]
            contour1 = np.hstack((contour1[:, 1].reshape(-1, 1), contour1[:, 0].reshape(-1, 1)))
            contour2 = np.hstack((contour2[:, 1].reshape(-1, 1), contour2[:, 0].reshape(-1, 1)))
            transformed1 = np.matmul(np.hstack((contour1, np.ones((len(contour1), 1)))), t1.T)
            transformed2 = np.matmul(np.hstack((contour2, np.ones((len(contour2), 1)))), t2.T)
            min_x1, min_x2 = transformed1[:, 0].min(), transformed2[:, 0].min()
            max_x1, max_x2 = transformed1[:, 0].max(), transformed2[:, 0].max()
            min_y1, min_y2 = transformed1[:, 1].min(), transformed2[:, 1].min()
            max_y1, max_y2 = transformed1[:, 1].max(), transformed2[:, 1].max()
            if (max_x2 - min_x1) * (min_x2 - max_x1) > 100 or (max_y2 - min_y1) * (min_y2 - max_y1) > 100:
                continue
            else:
                idx1, idx2 = \
                    data_preprocess.get_corresbounding(contour1, transformed2, t1)
                if len(idx1) <= 50:
                    continue
                else:
                    matching_set['source_ind'].append(idx1)
                    matching_set['target_ind'].append(idx2)
                    matching_set['GT_pairs'].append([current_nums + i, current_nums + k])

    return matching_set


'''------------------------------main-----------------------------'''

# fragment image path
data_path = "./"
sub_list = os.listdir(data_path)
'''get GT transformation'''
global matching_set

# save path
root = './'
count = []
if os.path.exists(root):
    with open(root, 'rb') as file:
        matching_set = pickle.load(file)
else:
    matching_set = {
        'full_pcd_all': [],
        'img_all': [],
        'extra_img': [],
        'shape_all': [],
        'GT_pairs': [],
        'source_ind': [],
        'target_ind': [],
        'overlap': [],
        'down_sample_pcd':[]
    }


count = 0
current_nums = len(matching_set['full_pcd_all'])
for n in range(len(sub_list)):
    img_path = os.path.join(data_path, sub_list[n])
    preprocess(img_path)
    current_nums = len(matching_set['full_pcd_all'])
    print(count)
    count += 1
    if count == 390:
        with open(root, 'wb') as file:
            pickle.dump(matching_set, file)
        break
