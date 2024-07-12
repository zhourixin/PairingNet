import time
import cv2
import torch
import pickle
import numpy as np
# from scipy.spatial.distance import cdist
import os

def feature_searching(result_matrix, gt_pair):
    """to get the topk searching result from score matrix"""
    # result_matrix = result_matrix + result_matrix.T
    sort_matrix = torch.sort(result_matrix, dim=-1, descending=True)

    idx = sort_matrix[1]
    idx = idx.numpy() #（3279，3279）  gt_pair（2370，2）
    l = []
    for i in range(len(gt_pair)):

        l.append(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))

    result = np.array(l).reshape(-1)

    top1 = (result < 1).sum() / len(l)
    top5 = (result < 5).sum() / len(l)
    top10 = (result < 10).sum() / len(l)
    top20 = (result < 20).sum() / len(l)

    return top1, top5, top10, top20

def searching(result_matrix, gt_pair):
    """to get the topk searching result from score matrix"""
    result_matrix = result_matrix + result_matrix.T
    sort_matrix = torch.sort(result_matrix, dim=-1, descending=True)

    idx = sort_matrix[1]
    idx = idx.numpy() #（3279，3279）  gt_pair（2370，2）
    l = []
    for i in range(len(gt_pair)):
        # if mins < len_all[i] <= maxs:
        l.append(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))

    result = np.array(l).reshape(-1)

    top1 = (result < 1).sum() / len(l)
    top5 = (result < 5).sum() / len(l)
    top10 = (result < 10).sum() / len(l)
    top20 = (result < 20).sum() / len(l)

    return top1, top5, top10, top20


def e_rmse(intersection_s_trans, intersection_t):
    """
    calculate the e_rmse score the matching result between two fragments.
    for calculating registration recall.
    """
    ermse = np.sqrt(sum(np.linalg.norm(intersection_s_trans-intersection_t, axis=-1)) / len(intersection_t))
    return ermse



def mean_chamfer_dist(pcd_s,pcd_t):
    dist_all=cdist(pcd_s,pcd_t) # cdist是一个包
    min_dist=[]
    for i in range(dist_all.shape[0]):
        min_dist.append(np.min(dist_all[i]))
    mean_dist=sum(min_dist)/len(min_dist)
    return mean_dist
def correspond_l2_dist(pcd_s,pcd_t):
    dist_sum=0
    for i in range(len(pcd_s)):
        point1=pcd_s[i]
        point2=pcd_t[i]
        dist_sum+=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist_sum/len(pcd_s)


def similarity_score(matrix, dilate, long, threshold1):
    """
    for calculating the score of similarity matrix.
    by searching the line in matrix with Hough transformation.
    """
    xy = np.where(dilate > 0) #获得矩阵中大于0的坐标点
    x, y = xy[0], xy[1]
    lines = cv2.HoughLines(dilate.astype(np.uint8), 1, np.pi/180, threshold=35, min_theta=np.pi*44/180, max_theta=np.pi*46/180)
    energy0 = 0  # sum(x)
    energy1 = 0  # np.trapz
    energy2 = 0  # np.exp(x)

    key_word = {}
    x_dic = {}
    y_dic = {}
    if lines is None:
        return torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
    for line in lines:
        conv = matrix[x, y]
        a = np.cos(line[0, 1])
        b = np.sin(line[0, 1])
        r = line[0, 0]

        if key_word.get(r//20):
            continue
        key_word[r//20] = r//20 #可以看作是精度
        mask = np.abs(y - (r - a * x) / b) <= 10 # 10是阈值

        conv = conv[mask]
        if len(conv) <= 10:
            continue
        filted = cv2.blur((conv*255).astype(np.uint8), ksize=(1, len(conv)//10), borderType=cv2.BORDER_CONSTANT).reshape(-1)/255
        # filted -= threshold1
        filted = filted[filted > threshold1]
        cur_energy1 = -np.log(filted).sum()
        cur_energy0 = np.sum(filted)
        cur_energy2 = np.exp(filted).sum()

        xy = np.vstack((x[mask], y[mask]))

        # to find the connected line at the edge of matrix
        if xy[0, 0] // 4 == 0: 
            y_dic[xy[1, 0] // 4] = [cur_energy0, cur_energy1, cur_energy2]   # record y on axis x
        if xy[1, -1] // 4 == long[1]:
            x_dic[xy[0, -1] // 4] = [cur_energy0, cur_energy1, cur_energy2]  # record x on axis y

        if xy[0, -1] // 4 == long[0] // 4:  # if existed on axis y
            if x_dic.get(xy[1, -1] // 4):
                cur_energy0 += x_dic[xy[1, -1]//2][0]
                cur_energy1 += x_dic[xy[1, -1]//2][1]
                cur_energy2 += x_dic[xy[1, -1]//2][2]

        if xy[1, 0] // 4 == long[1] // 4:  # if existed on axis x
            if y_dic.get(xy[0, 0] // 4):
                cur_energy0 += y_dic[xy[0, 0] // 4][0]
                cur_energy1 += y_dic[xy[0, 0] // 4][1]
                cur_energy2 += y_dic[xy[0, 0] // 4][2]

        if cur_energy0 > energy0:
            energy0 = cur_energy0
        if cur_energy1 > energy1:
            energy1 = cur_energy1
        if cur_energy2 > energy2:
            energy2 = cur_energy2

    return torch.tensor([energy0]), torch.tensor([energy1]), torch.tensor([energy2])

