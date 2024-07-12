import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
import cv2
import torch
import ransac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utilz import rigid_transform_2d


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.4f},{b:.4f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


class Visualization(object):
    def __init__(self, gt_matrix, conv_matrix, pcd_s, pcd_t, img_s, img_t, ind_s_origin, ind_t_origin, s_pcd_origin, t_pcd_origin, conv_threshold):
        """
        all input in type of tensor/numpy, cpu
        :param gt_matrix: type = float, tensor
        :param conv_matrix: type = float, tensor
        :param pcd_s: type = float, tensor
        :param pcd_t: type = float, tensor
        :param img_s: type = float, numpy
        :param img_t: type = float, numpy
        """
        '''initial define pcd & img & conv_matrix'''
        self.pcd_s, self.pcd_t = pcd_s, pcd_t
        self.img_s = img_s
        self.img_t = img_t
        self.matrix_pre = conv_matrix
        self.gt_matrix = gt_matrix

        '''get predict initial'''
        idx = np.where(conv_matrix > conv_threshold)
        self.source_ind = idx[0]
        self.tar_ind = idx[1]
        self.pcd_s_inter = pcd_s[self.source_ind].reshape(-1, 2)
        self.pcd_t_inter = pcd_t[self.tar_ind].reshape(-1, 2)

        '''get origin gt initial'''
        self.source_ind_gt_origin = ind_s_origin
        self.tar_ind_gt_origin = ind_t_origin
        self.pcd_s_inter_gt_origin = s_pcd_origin[self.source_ind_gt_origin].reshape(-1, 2)
        self.pcd_t_inter_gt_origin = t_pcd_origin[self.tar_ind_gt_origin].reshape(-1, 2)
        
        '''get gt initial'''
        idx = np.where(gt_matrix > conv_threshold)
        self.source_ind_gt = idx[0]
        self.tar_ind_gt = idx[1]
        self.pcd_s_inter_gt = pcd_s[self.source_ind_gt]
        self.pcd_t_inter_gt = pcd_t[self.tar_ind_gt]
        self.GT_transformation = rigid_transform_2d(self.pcd_s_inter_gt, self.pcd_t_inter_gt)

    def get_corresponding(self, path):
        """show corresponding"""
        plt.figure()
        plt.scatter(self.pcd_s[:, 0], self.pcd_s[:, 1])
        plt.scatter(self.pcd_t[:, 0], self.pcd_t[:, 1])
        for i in range(len(self.pcd_t_inter_gt)):
            plt.plot(
                [self.pcd_s_inter_gt[i][0], self.pcd_t_inter_gt[i][0]],
                [self.pcd_s_inter_gt[i][1], self.pcd_t_inter_gt[i][1]],
                c='green')

        for i in range(len(self.pcd_t_inter)):
            plt.plot(
                [self.pcd_s_inter[i][0], self.pcd_t_inter[i][0]],
                [self.pcd_s_inter[i][1], self.pcd_t_inter[i][1]],
                c='red')
        plt.savefig(path)
        plt.close()

    def get_transformation(self):
        """get transformation use RANSAC"""

        if len(self.pcd_s_inter) < 20:
            return None, None
        transformation, pairs, _ = ransac.ransac_matchV2(
            self.pcd_s_inter, self.pcd_t_inter)
        
        return transformation, pairs

    def get_img(self, path, transformation):
        """show img"""
        if transformation is not None:
            transformation = np.delete(transformation[:2], 2, axis=-1)
            transformation[0][2] += 80
            transformation[1][2] += 80
            '''show img'''
            img_s = self.img_s.transpose(1, 0, 2)
            img_t = self.img_t.transpose(1, 0, 2)
            # img_s, img_t = 255-img_s, 255-img_t
            w1, h1 = int(max(self.pcd_s[:, 0]) - min(self.pcd_s[:, 0])), int(
                max(self.pcd_s[:, 1]) - min(self.pcd_s[:, 1]))
            w2, h2 = int(max(self.pcd_t[:, 0]) - min(self.pcd_t[:, 0])), int(
                max(self.pcd_t[:, 1]) - min(self.pcd_t[:, 1]))
            edge = int(max(h1 + h1, w1 + w2))
            changed_s = np.matmul(np.hstack((self.pcd_s, np.ones((len(self.pcd_s), 1)))), transformation.T)
            merge = np.vstack((changed_s, self.pcd_t))
            x_mid = max(merge[:, 0]) + min(merge[:, 0])
            x_mid = x_mid // 2
            y_mid = max(merge[:, 1]) + min(merge[:, 1])
            y_mid = y_mid // 2
            center_m = np.array([[1, 0, edge // 2 - x_mid], [0, 1, edge // 2 - y_mid]], dtype=np.float32)
            img2_t = cv2.warpAffine(img_t, center_m, (edge, edge))
            new_m = np.matmul(center_m, np.vstack((transformation, np.array([[0, 0, 1]]))))
            t_img = cv2.warpAffine(img_s, new_m, (edge, edge))
            result = t_img + img2_t

            result = result.transpose(1, 0, 2)
            cv2.imwrite(path, result)

        return

    def get_gt_img(self, path):
        transformation = self.GT_transformation
        img_s = self.img_s.transpose(1, 0, 2)
        img_t = self.img_t.transpose(1, 0, 2)

        w1, h1 = int(max(self.pcd_s[:, 0]) - min(self.pcd_s[:, 0])), int(
            max(self.pcd_s[:, 1]) - min(self.pcd_s[:, 1]))
        w2, h2 = int(max(self.pcd_t[:, 0]) - min(self.pcd_t[:, 0])), int(
            max(self.pcd_t[:, 1]) - min(self.pcd_t[:, 1]))
        edge = int(max(h1 + h1, w1 + w2))
        changed_s = np.matmul(np.hstack((self.pcd_s, np.ones((len(self.pcd_s), 1)))), transformation.T)
        merge = np.vstack((changed_s, self.pcd_t))
        x_mid = max(merge[:, 0]) + min(merge[:, 0])
        x_mid = x_mid // 2
        y_mid = max(merge[:, 1]) + min(merge[:, 1])
        y_mid = y_mid // 2
        center_m = np.array([[1, 0, edge // 2 - x_mid], [0, 1, edge // 2 - y_mid]], dtype=np.float32)
        img2_t = cv2.warpAffine(img_t, center_m, (edge, edge))
        new_m = np.matmul(center_m, np.vstack((transformation, np.array([[0, 0, 1]]))))
        t_img = cv2.warpAffine(img_s, new_m, (edge, edge))
        result = t_img + img2_t

        result = result.transpose(1, 0, 2)
        cv2.imwrite(path, result)

    @staticmethod
    def weight_visualize(path, img, pcd, w_s):
        """ """

        w_s = np.sum(np.where(w_s > 0.5, 1/64, 0), axis=1)

        w_s = w_s - np.min(w_s)
        w_s = w_s / np.max(w_s)

        w_s = w_s.reshape(-1)
        pcd = pcd[(pcd != 0).any(axis=-1)]
        c_map = plt.get_cmap('winter')

        new_cmap = truncate_colormap(c_map, 0, 1, n=10000)
        c = new_cmap(w_s, bytes=True)[:, :3]
        for i in range(len(pcd)):
            cv2.circle(img, tuple(pcd[i].astype(np.int)), 1, tuple(map(int, c[i])), -1)


        return img.transpose(1, 0, 2)

    
    @staticmethod
    def weight_visualize2(path, img, pcd, w_s):
        """ """

        pcd = pcd[(pcd != 0).any(axis=-1)]
        c_map = plt.get_cmap('coolwarm')

        new_cmap = truncate_colormap(c_map,-2.5, 4.5, n=100)
        c = new_cmap(w_s, bytes=True)[:, :3]
        for i in range(len(pcd)):
            cv2.circle(img, tuple(pcd[i].astype(np.int)), 2, tuple(map(int, c[i])), -1)

        return img





