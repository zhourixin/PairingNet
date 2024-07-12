import os
import sys
import cv2
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense() # D^-0.5AD^0.5


def get_adjacent(boundary, sparse=False, source=True):
    """
    The input is a set of contour points, and the function will get an adjacency matrix constructed
    from the set of points.
    :param boundary: type = Ndarray.
    :param sparse: type = bool
    :return: an adjacency matrix.
    """
    n = len(boundary)
    adjacent_matrix = np.zeros((n, n))
    if source:
        for i in range(n):
            # adjacent_matrix[i][[i - 3, i - 2, i - 1, (i + 1) % n, (i + 2) % n, (i + 3) % n]] = np.array([13, 12, 11, 9, 8, 7])
            adjacent_matrix[i][[i - 3, i - 2, i - 1, (i + 1) % n, (i + 2) % n, (i + 3) % n]] = np.array(
                [10])
            # adjacent_matrix[i][[(i+1) % n]] = np.array([9])
            # adjacent_matrix[i][[i - 1]] = np.array([11])
    else:
        for i in range(n):
            # adjacent_matrix[i][[i - 3, i - 2, i - 1, (i + 1) % n, (i + 2) % n, (i + 3) % n]] = np.array([7, 8, 9, 11, 12, 13])
            adjacent_matrix[i][[i - 3, i - 2, i - 1, (i + 1) % n, (i + 2) % n, (i + 3) % n]] = np.array(
                [10])
            # adjacent_matrix[i][[(i + 1) % n]] = np.array([11])
            # adjacent_matrix[i][[i-1]] = np.array([9])

    adj = adjacent_matrix + np.eye(len(adjacent_matrix)) * 10
    if sparse:
        i = np.vstack((adj[0], adj[1]))
        v = torch.ones(len(i[0]))
        adj = torch.sparse.FloatTensor(torch.from_numpy(i), v, adjacent_matrix.shape)
    return adj


def get_corresbounding(source_pcd, target_pcd, matrix):
    """
    to Get the complete dataset parameters, including the intersection of the two contours,
    the corresponding point subscript of the intersection, and the corresponding point matrix.
    :param source_pcd: type = Ndarray
    :param target_pcd: type = Ndarray
    :param matrix: transformation of source point set. type = Ndarray
    :return: source intersection, target intersection, source subscript, target subscript, GT matrix.
    """
    source_pcd_trans = np.matmul(np.hstack((source_pcd, np.ones((len(source_pcd), 1)))), matrix.T)
    # source_pcd_trans = np.hstack((source_pcd_trans[:, 1].reshape(-1, 1), source_pcd_trans[:, 0].reshape(-1, 1)))
    intersection_m = np.matmul(source_pcd_trans / np.linalg.norm(source_pcd_trans, axis=-1).reshape(-1, 1),
                               (target_pcd / np.linalg.norm(target_pcd, axis=-1).reshape(-1, 1)).transpose())
    cores_m = torch.from_numpy(intersection_m)
    cores_source_max, cores_tar_max = torch.max(cores_m, dim=1)[0], torch.max(cores_m, dim=0)[0]
    cores_source_ind, cores_tar_ind = \
        torch.arange(0, len(source_pcd))[cores_source_max > 0.99], \
        torch.arange(0, len(target_pcd))[cores_tar_max > 0.99]

    tree = KDTree(target_pcd, leaf_size=2)
    source_ind, target_ind = [], []
    ddd = 99999
    for i in cores_source_ind:
        d, idx = tree.query(source_pcd_trans[i].reshape(-1, 2), k=1)
        if d < ddd:
            ddd = d
        if d < 2:  # 30 for preprocessing OBI dataset
            source_ind.append(i)
            target_ind.append(idx[0, 0])
    cores_source_ind, cores_tar_ind = np.array(source_ind), np.array(target_ind)

    return cores_source_ind, cores_tar_ind


class MyDataSet(Dataset):
    def __init__(self, GT_config):
        super(MyDataSet, self).__init__()
        self.inputs = {
            'pcd_all': [],
            'full_pcd_all': [],
            'img_all': [],
            'adj_all_s': [],
            'adj_all_t': [],
            'shape_all': [],
            'length_all': [],
            'GT_pairs': GT_config['GT_pairs'],
        }
        self.model = GT_config['train_model']
        n = len(GT_config['pcd_all'])
        pair_n = len(GT_config['GT_pairs'])
        # train_end = int(n * 0.9)
        if GT_config['train_model'] == 'train':
            start = 0
            end = n
            n = n
        elif GT_config['train_model'] == 'test':
            start = 0
            end = 200
            n = n
        else:
            start = 0
            end = n
            n = n
        c = GT_config['channel']
        long = list(map(lambda x: len(x), GT_config['full_pcd_all']))
        short = list(map(lambda x: len(x), GT_config['pcd_all']))
        maxs = max(short)
        full_max = max(long)
        shape_all = np.array(GT_config['shape_all'])
        shape_all = torch.from_numpy(shape_all)[:, :2]
        height_max = max(shape_all[:, 0].max(), shape_all[:, 1].max())
        width_max = height_max
        self.inputs['pcd_all'] = torch.zeros((n, maxs, 2))
        self.inputs['full_pcd_all'] = torch.zeros((n, full_max, 2))
        self.inputs['img_all'] = torch.zeros(n, c, 224, 224)
        self.inputs['adj_all_s'] = torch.zeros((n, maxs, maxs))
        self.inputs['adj_all_t'] = torch.zeros((n, maxs, maxs))
        self.inputs['length_all'] = torch.zeros((n, 1), dtype=torch.int)
        for i in range(n):
            '''load_basic'''
            pcd = GT_config['pcd_all'][i]
            full_pcd = GT_config['full_pcd_all'][i]
            len_pcd = len(pcd)

            '''pcd'''
            self.inputs['pcd_all'][i][0:short[i]] = torch.from_numpy(pcd)
            self.inputs['full_pcd_all'][i][0:long[i]] = torch.from_numpy(full_pcd)
            # self.inputs['pcd_all'][i][len_pcd % maxs] = torch.from_numpy(pcd)[0]
            # self.inputs['pcd_all'][i][-1] = torch.from_numpy(pcd)[-1]
            self.inputs['length_all'][i] = len_pcd

            '''adjacency matrix'''
            adj_s = get_adjacent(pcd, source=True)
            adj_t = get_adjacent(pcd, source=False)
            adj_s = normalize_adj(adj_s)
            adj_t = normalize_adj(adj_t)
            self.inputs['adj_all_s'][i][0:len_pcd, 0:len_pcd] = torch.from_numpy(adj_s)
            self.inputs['adj_all_t'][i][0:len_pcd, 0:len_pcd] = torch.from_numpy(adj_t)

            '''img'''
            temp_empty_img = np.zeros((height_max, width_max, c))
            temp_empty_img[:shape_all[i][1], :shape_all[i][0]] = GT_config['img_all'][i]
            source_img = cv2.resize(
                temp_empty_img, (224, 224), interpolation=cv2.INTER_NEAREST).reshape(224, 224, c)
            img = torch.from_numpy(source_img.transpose((2, 0, 1)))
            self.inputs['img_all'][i] = img
            if i % 100 == 0:
                print('{} complete'.format(i))
        self.inputs['shape'] = [height_max, height_max]

        '''处理GT对'''
        if GT_config['train_model'] != 'try_anything':
            self.inputs['mask_all'] = torch.zeros((pair_n, maxs, maxs))
            self.inputs['l_mask'] = torch.zeros((pair_n, maxs, maxs))
            for i in range(pair_n):
                source_ind = GT_config['source_ind'][i + start]
                target_ind = GT_config['target_ind'][i + start]
                l1, l2 = GT_config['GT_pairs'][i]
                mask = torch.zeros((short[l1], short[l2]), dtype=torch.bool)
                mask[source_ind, target_ind] = True
                len_s, len_t = self.inputs['length_all'][l1], self.inputs['length_all'][l2]
                self.inputs['mask_all'][i][0:len_s, 0:len_t] = mask
                self.inputs['l_mask'][i][0:len_s, 0:len_t] = 1
                if i % 100 == 0:
                    print('{} complete'.format(i))
        else:
            triu_idx = np.triu_indices(n)
            triu_idx = np.vstack((triu_idx[0], triu_idx[1])).transpose()
            self.inputs['GT_pairs'] = torch.from_numpy(triu_idx)

    def __len__(self):
        return len(self.inputs['GT_pairs'])

    def __getitem__(self, idx):
        self.inputs['GT_pairs'] = np.array(self.inputs['GT_pairs'])
        if self.model != 'matching':
            idx_s, idx_t = self.inputs['GT_pairs'][idx]
            # temp_list = []
            # temp_list.append(idx_s)
            # temp_list.append(idx_t)
            # temp_ind = idx-1
            # while self.inputs['GT_pairs'][temp_ind % len(self)][0] == idx_s:
            #     temp_list.append(self.inputs['GT_pairs'][temp_ind % len(self)][1])
            #     temp_ind -= 1
            # temp_ind = idx + 1
            # while self.inputs['GT_pairs'][temp_ind % len(self)][0] == idx_s:
            #     temp_list.append(self.inputs['GT_pairs'][temp_ind % len(self)][1])
            #     temp_ind += 1
            # idx_neg = np.random.randint(0, len(self.inputs['pcd_all']))
            # if idx_neg in temp_list:
            #     idx_neg = np.random.randint(0, len(self.inputs['pcd_all']))

            return \
                (self.inputs['pcd_all'][idx_s], self.inputs['pcd_all'][idx_t]), \
                (self.inputs['adj_all_s'][idx_t], self.inputs['adj_all_t'][idx_t]), \
                (self.inputs['l_mask'][idx], self.inputs['mask_all'][idx]), \
                (self.inputs['img_all'][idx_s], self.inputs['img_all'][idx_t]), \
                (self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t])

        else:
            idx_s, idx_t = self.inputs['GT_pairs'][idx][0], self.inputs['GT_pairs'][idx][1]
            return \
                (self.inputs['pcd_all'][idx_s], self.inputs['pcd_all'][idx_t]), \
                (self.inputs['adj_all_s'][idx_t], self.inputs['adj_all_t'][idx_t]), \
                (self.inputs['img_all'][idx_s], self.inputs['img_all'][idx_t]), \
                (self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t])











