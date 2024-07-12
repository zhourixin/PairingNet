import cv2
import time
import torch
import tqdm
import numpy as np
from torch.utils.data import Dataset
from encoder import pre_encoder1, pre_encoder2, pre_encoder3, img_patch_encoder
from torchvision import transforms as tf
import math
from PIL import Image

def get_area(poly, max_length):
    empty = np.zeros((max_length, max_length), dtype=np.uint8)
    color = 255
    mask = cv2.fillPoly(empty, [poly], (color))
    mask = (mask == color).sum()

    return mask


def get_adjacent(boundary, max_len, k=1):
    """
    The input is a set of contour points, and the function will get an adjacency matrix constructed
    from the set of points.
    :param boundary: type = Ndarray.
    :param sparse: type = bool
    :return: an adjacency matrix.
    """
    n = len(boundary)
    adjacent_matrix = np.eye(n)
    temp = np.eye(n)
    for i in range(k):
        adjacent_matrix += np.roll(temp, i + 1, axis=0)
        adjacent_matrix += np.roll(temp, -i - 1, axis=0)
    temp = np.zeros((max_len, max_len))
    temp[:n, :n] = adjacent_matrix
    return torch.from_numpy(temp).to_sparse()

def get_adjacent2(boundary, max_len, k=1):
    """
    The input is a set of contour points, and the function will get an adjacency matrix constructed
    from the set of points.
    :param boundary: type = Ndarray.
    :param sparse: type = bool
    :return: an adjacency matrix.
    返回非稀疏矩阵
    """
    n = len(boundary)
    adjacent_matrix = np.eye(n)
    temp = np.eye(n)
    for i in range(k):
        adjacent_matrix += np.roll(temp, i + 1, axis=0)
        adjacent_matrix += np.roll(temp, -i - 1, axis=0)
    temp = np.zeros((max_len, max_len))
    temp[:n, :n] = adjacent_matrix

    return torch.from_numpy(temp)

def generate_tensor(n, max_length):

    tensor = torch.zeros((max_length, max_length), dtype=torch.bool)
    tensor[:n, :n] = True

    return tensor 


class MyDataSet(Dataset):
    def __init__(self, GT_config, args):
        super(MyDataSet, self).__init__()
        self.inputs = {
            'full_pcd_all': [],
            'img_all': [],
            'c_input': [],
            't_input': [],
            'GT_pairs': GT_config['GT_pairs'],
            "att_mask_s":[],
            "att_mask_t":[]
        }
        # initial parameters
        self.model = GT_config['model_type']  # train, test, matching
        patch_size = GT_config['patch_size']  # 3x3, 7x7, 11x11
        c_model = GT_config['c_model']  # l, io, ilo
        n = len(GT_config['img_all'])  # nums of fragments
        pair_n = len(GT_config['GT_pairs'])  # nums of gt pairs
    
        c = GT_config['channel']
        trans = tf.Compose([
            tf.ToTensor(),
            tf.Resize(224),
            tf.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        # get max point nums
        self.long = list(map(lambda x: len(x), GT_config['full_pcd_all'])) #每个轮廓的长度

        max_points = args.max_length
        self.max_points = max_points

        # get max img shape
        shape_all = np.array(GT_config['shape_all'])
        shape_all = torch.from_numpy(shape_all)[:, :2]
        height_max = 1319  # length of max length
        width_max = height_max
        mid_area = height_max ** 2

        print("Update adjacency matrix")
        for i in tqdm.trange(n):

            GT_config['adj_all'][i] = get_adjacent2(GT_config['full_pcd_all'][i], max_points, k = 8)

        # initial inputs
        self.inputs['full_pcd_all'] = torch.zeros((n, max_points, 2))  # input contours
        self.inputs['c_input'] = torch.zeros((n, max_points, patch_size, patch_size))  # input patches
        self.inputs['t_input'] = torch.zeros((n, max_points, 3, patch_size, patch_size))  # input patches
        self.inputs['img_all'] = torch.zeros((n, c, 224, 224))  # input image
        self.inputs['adj_all'] = GT_config['adj_all']
        self.inputs['factor'] = torch.zeros(n)  

        #  deal with each inputs of fragments
        print('dealing with fragments')
        for i in tqdm.trange(n):
      
            '''points'''
            full_pcd = GT_config['full_pcd_all'][i]
            self.inputs['full_pcd_all'][i][0:self.long[i]] = torch.from_numpy(full_pcd)


            '''img_ori'''
            temp_empty_img = np.zeros((height_max, width_max, c), dtype=np.uint8) 
            temp_empty_img[:shape_all[i][1], :shape_all[i][0]] = cv2.cvtColor(GT_config['img_all'][i],
                                                                                          cv2.COLOR_BGR2RGB)                                                                            
            # resize 2 224 x 224
            new_img = trans(temp_empty_img)
            self.inputs['img_all'][i] = new_img
            self.inputs['factor'][i] = 1

            # input patches
            img = cv2.cvtColor(GT_config['img_all'][i], cv2.COLOR_BGR2RGB) # 311，298，3
            img = np.pad(img, ((0, 20), (0, 20), (0, 0)), 'constant', constant_values=(0, 0)) # 331，318，3
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255 # 1,3,331,318
            t_input = img_patch_encoder(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size)
            self.inputs['t_input'][i] = t_input[0]
            img = (GT_config['img_all'][i] != 0).all(-1)  # get extracted template
            img = np.pad(img, ((0, 20), (0, 20)), 'constant', constant_values=(0, 0))
            img = torch.from_numpy(img).float().unsqueeze(0) #这一步之后，有像素的地方都是1，无像素的地方都是0

            if c_model == 'l':  # only contour line
                c_input = pre_encoder1(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size) # 1,2778,7,7
            elif c_model == 'io':  # Interior and exterior of contour
                c_input = pre_encoder2(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size)
            else:  # Interior, exterior and contour
                c_input = pre_encoder3(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size)
            self.inputs['c_input'][i] = c_input[0]
        self.inputs['shape'] = [height_max, height_max]

        # points normalization
        self.inputs['full_pcd_all'] = self.inputs['full_pcd_all'] / (height_max / 2.) - 1

        #  deal with correctly matched pairs into matrix
        if GT_config['model_type'] != 'searching':
            print("dealing with gt pairs into matrix")
            self.inputs['mask_all'] = []
            self.inputs['att_mask_s'] = []
            self.inputs['att_mask_t'] = []
            for i in tqdm.trange(pair_n):
                source_intersection_ind = GT_config['source_ind'][i]
                target_intersection_ind = GT_config['target_ind'][i]
                mask = torch.zeros((max_points, max_points), dtype=torch.bool)
                mask[source_intersection_ind, target_intersection_ind] = True

                self.inputs['mask_all'].append(mask)
                self.inputs['att_mask_s'].append(torch.zeros((1,1)))
                self.inputs['att_mask_t'].append(torch.zeros((1,1)))

            self.inputs['mask_all'] = torch.stack(self.inputs['mask_all'], 0)
            self.inputs['att_mask_s'] = torch.stack(self.inputs['att_mask_s'], 0)
            self.inputs['att_mask_t'] = torch.stack(self.inputs['att_mask_t'], 0)

    def __len__(self):
        if self.model == 'searching' or self.model == 'stage1':
            return len(self.inputs['img_all'])
        elif  self.model == 'train' or self.model == 'test':
            return len(self.inputs['GT_pairs'])

    def __getitem__(self, idx):
        self.inputs['GT_pairs'] = np.array(self.inputs['GT_pairs'])
        if self.model == 'matching_train' or self.model == 'matching_test':
            idx_s, idx_t = self.inputs['GT_pairs'][idx]
            full_s, full_t = self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t]
            return \
                (self.inputs['mask_all'][idx], self.long[idx_s], self.long[idx_t],idx_s, idx_t), \
                (self.inputs['img_all'][idx_s], self.inputs['img_all'][idx_t]), \
                (full_s, full_t), \
                (self.inputs['c_input'][idx_s], self.inputs['c_input'][idx_t]), \
                (self.inputs['t_input'][idx_s], self.inputs['t_input'][idx_t]), \
                (self.inputs['adj_all'][idx_s], self.inputs['adj_all'][idx_t]), \
                (self.inputs['factor'][idx_s], self.inputs['factor'][idx_t]), \
                (self.inputs['att_mask_s'][idx], self.inputs['att_mask_t'][idx])


        elif self.model == 'save_stage1_feature' :
            return self.inputs['full_pcd_all'][idx], self.inputs['img_all'][idx], self.inputs['t_input'][idx], \
                   self.inputs['adj_all'][idx], self.inputs['factor'][idx], self.inputs['c_input'][idx]


class MyDataSet_searching(Dataset):
    def __init__(self, stage1_feature, args):
        self.stage1_feature = stage1_feature["saved_feature"]
        self.GT_pairs = stage1_feature["GT_pairs"]
        self.full_pcd = stage1_feature["full_pcd"]
        self.model = args.model_type
        self.adj = []
        self.inputs = {
            'full_pcd_all': [],
            'source_ind':stage1_feature["source_ind"],
            'target_ind':stage1_feature["target_ind"]
        }

        n = len(self.full_pcd)
        max_points = args.max_length
        self.inputs['full_pcd_all'] = torch.zeros((n, max_points, 2))
        self.long = list(map(lambda x: len(x), self.full_pcd))
        height_max = 1319

        self.inputs['full_pcd_all'] = self.inputs['full_pcd_all'] / (height_max / 2.) - 1

    
    def __len__(self):
        if self.model == 'stage2':
            return len(self.GT_pairs)
        elif self.model == 'stage2_searching':
            return len(self.stage1_feature)

    def __getitem__(self, idx):
        if self.model == 'stage2':
            idx_s, idx_t = self.GT_pairs[idx]
            return (self.stage1_feature[idx_s], self.stage1_feature[idx_t], idx_s, idx_t, self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t]), \
                  (self.long[idx_s], self.long[idx_t])

        elif self.model == 'stage2_searching':
            return self.stage1_feature[idx], self.inputs['full_pcd_all'][idx]