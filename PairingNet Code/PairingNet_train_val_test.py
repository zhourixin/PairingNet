import __init__
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import torch
import math
import random
import pickle
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import numpy as np
from glob import glob
from tqdm import tqdm
from hausdorff import hausdorff_distance
from utils.loss import FocalLoss
from utils.evaluation import e_rmse
from utils.utilz import affine_transform
from utils import pipeline, config, data_preprocess, visualization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import re
global c
from torch.utils.data.distributed import DistributedSampler
from utils.infornce_loss import InfoNCE
import torch.nn.functional as F
import numpy as np
import pytorch_warmup as warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary
from utils import calute_NDCG 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

def pad_tensor(t1, t2):
    
    s1 = t1.shape
    s2 = t2.shape
    if s1[-1] >= s2[-1]:
        size_tensor = torch.full((s1[0], 1), s1[-1], dtype=torch.int)
        padded_t1 = torch.cat([t1, size_tensor], dim=-1)
        return padded_t1
    else:
        padding = s2[-1] - s1[-1]
        padded_t1 = torch.cat([t1, torch.zeros(*s1[:-1], padding, dtype=torch.int)], dim=-1)
        size_tensor = torch.full((s1[0], 1), s1[-1], dtype=torch.int)
        padded_t1 = torch.cat([padded_t1, size_tensor], dim=-1)
        return padded_t1

def unpad_tensor(padded_t1):
    original_size = int(padded_t1[0, -1])
    padded_t1 = padded_t1[:, :-1]
    unpadded_t1 = torch.narrow(padded_t1, -1, 0, original_size)
    return unpadded_t1

class Train_model(object):
    def __init__(self, net, args, temperature, case_name):
        """"""
        '''initial tensorboard'''
        self.log_save_path = EXP_path+'/EXP/{}/summary'.format(case_name)
        self.checkpoint_path = EXP_path+'/EXP/{}/checkpoint'.format(case_name)
        self.case_name = case_name
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_save_path)

        
        '''set training dataset'''
        print('set training dataset')
        self.train_data, _ = self.set_dataset(args.train_set, args)
        self.train_loader = DataLoader(self.train_data, args.matching_batch_size, num_workers=0,shuffle=True)

        '''set test set in training model'''
        self.valid_data, _ = self.set_dataset(args.valid_set, args)
        self.valid_loader = DataLoader(self.valid_data, 1, num_workers=0, shuffle=False)
        

        '''set training model'''
        print('set training model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.models = torch.nn.DataParallel(self.models)

        self.optimizer = torch.optim.Adam(self.models.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epoch)
        self.epoch = args.epoch
        self.args = args

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + '/checkpoint_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({  # 'state': torch.cuda.get_rng_state_all(),
                'epoch': epoch,
                'model_state_dict': self.models.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + '/checkpoint_{}.tar'.format(int(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return epoch

    @staticmethod
    def get_pad_mask(mask_para):
        """
        input points in each fragments are padded to a fixed nums.
        padded mask denotes the padded part in similarity matrix which
        calculated by source and target point feature.
        """
        bs, maxs, _ = mask_para[0].shape
        pad_mask = torch.zeros((bs, maxs, maxs), dtype=torch.bool)
        for m in range(bs):
            a = mask_para[2][m]
            b = mask_para[1][m]
            pad_mask[m][:, mask_para[2][m]:] = True
            pad_mask[m][mask_para[1][m]:, :] = True

        return pad_mask

    def get_concat_adj(adj, max_len):
        device = adj.device
        temp_adj = torch.zeros((2, 0), dtype=torch.int).to(device)
        for i in range(len(adj)):
            b = torch.nonzero(adj[i]).transpose(0, 1)
            # a = adj[i].coalesce().indices() #(2,8602)
            temp_adj = torch.hstack((temp_adj, b + i * max_len))

        return temp_adj
    @staticmethod
    def get_concat_adj2(adj, max_len):
        device = adj.device
        temp_adj = torch.zeros((2, 0), dtype=torch.int).to(device)
        for i in range(len(adj)):
            b = torch.nonzero(adj[i]).transpose(0, 1)
            # a = adj[i].coalesce().indices() #(2,8602)
            temp_adj = torch.hstack((temp_adj, b + i * max_len))

        return temp_adj
    

    def get_similarity_matrix(self, feature1, feature2, pad_mask):
        similarity_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1)) / self.temperature
        similarity_matrix[pad_mask] -= 1e9  # give a very small value to the padded part for softmax operation
        s_i = torch.softmax(similarity_matrix, dim=1)  # row softmax
        s_j = torch.softmax(similarity_matrix, dim=-1)  # column softmax
        similarity_matrix = torch.multiply(s_i, s_j)

        return similarity_matrix

    @staticmethod
    def set_dataset(data_path, args):
        with open(data_path, 'rb') as gt_file:
            gt_config = pickle.load(gt_file)

        gt_config['model_type'] = args.model_type
        gt_config['channel'] = args.channel
        gt_config['c_model'] = args.c_model
        gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyDataSet(gt_config, args)
        return dataset, gt_config

    def train_start(self):
        device = self.device
        '''start training'''
        print('start!!!')
        epoch = self.load_checkpoint()
        # epoch = 0
        min_loss = torch.inf
        for i in range(self.epoch):
            if i > self.epoch:
                break
            i += epoch
            loss_m_all = torch.zeros([0])
            v_loss_np_all = torch.zeros([0])
            p_all = torch.zeros([0])
            v_p_all = torch.zeros([0])
            self.models.train()
            self.models.requires_grad_(True)

            for _, (mask_para, imgs, pcd, c_input, t_input, adjs, factors, att_mask) in enumerate(tqdm(self.train_loader)):
                max_point_nums = len(pcd[0][0])
                adj_s = self.get_concat_adj2(adjs[0], max_point_nums)
                adj_t = self.get_concat_adj2(adjs[1], max_point_nums)
                # adj_s = adj_s.to(device)

                source_input = {
                    'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                    'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device), "att_mask":adjs[0].to(device)
                }

                target_input = {
                    'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                    'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device), "att_mask":adjs[1].to(device)
                }


                pad_mask = self.get_pad_mask(mask_para).to(device)  # mark the padded part in similarity matrix
                gt_mask = mask_para[0].to(device)  # mark the gt corresponding in similarity matrix

                feature_s, _, w_s = self.models(source_input) # 15,2778,64
                feature_t, _, w_t = self.models(target_input)
                similarity_matrix = self.get_similarity_matrix(feature_s, feature_t, pad_mask) #bs, n, n
                '''matching loss'''
                loss_fn = FocalLoss()
                pad_mask = torch.add(pad_mask, gt_mask)  # padded mask with gt label.
                loss_np, loss_p = loss_fn(similarity_matrix, gt_mask, pad_mask)


                self.optimizer.zero_grad()
                loss_np.backward()
                self.optimizer.step()
                loss_m_all = torch.cat((loss_m_all, loss_np.detach().cpu().view(-1)))
                p_all = torch.cat((p_all, loss_p.cpu().view(-1)))

            self.scheduler.step()
            self.writer.add_scalar('train_loss', loss_m_all.mean(), i)

            if (i + 1) % 2 != 0:
                continue

            '''validation'''
            self.models.eval()
            self.models.requires_grad_(False)
            for _, (mask_para, imgs, pcd, c_input, t_input, adjs, factors, att_mask) in enumerate(tqdm(self.valid_loader)):
                max_point_nums = len(pcd[0][0])
                adj_s = self.get_concat_adj2(adjs[0], max_point_nums)
                adj_t = self.get_concat_adj2(adjs[1], max_point_nums)

                source_input = {
                    'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                    'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device), "att_mask":att_mask[0].to(device)
                }

                target_input = {
                    'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                    'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device), "att_mask":att_mask[1].to(device)
                }

                pad_mask = self.get_pad_mask(mask_para).to(device)
                gt_mask = mask_para[0].to(device)
                feature_s, _, w_s = self.models(source_input)
                feature_t, _, w_t = self.models(target_input)
                similarity_matrix = self.get_similarity_matrix(feature_s, feature_t, pad_mask)
                '''matching loss'''
                loss_fn = FocalLoss()
                pad_mask = torch.add(pad_mask, gt_mask)  # padded mask with gt label.
                v_loss_np, v_loss_p = loss_fn(similarity_matrix, gt_mask, pad_mask)

                v_loss_np_all = torch.cat((v_loss_np_all, v_loss_np.cpu().view(-1)))
                v_p_all = torch.cat((v_p_all, v_loss_p.cpu().view(-1)))

            self.writer.add_scalar('valid_loss', v_loss_np_all.mean(), i)
            self.writer.add_scalar('valid_positive_loss', v_p_all.mean(), i)
            means_all = v_p_all.mean()
            # self.save_checkpoint(i)
            if means_all < min_loss:
                for path in glob(EXP_path+'/EXP/{}'.format(self.case_name) + '/val_min=*'):
                    os.remove(path)
                min_loss = means_all.clone()
                np.save(EXP_path+'/EXP/{}'.format(self.case_name) + '/val_min={}-{}'.format(i, min_loss), [i, min_loss])
                self.save_checkpoint(i)
                # torch.save(models.state_dict(), weight_save_path[:-4]+'({})'.format(round(float(min_loss), 2))+'.pth')

            print('epoch = {}, match_loss = {}, loss_p = {}, v_match_loss = {}, v_loss_p = {}'.format(
                i, loss_m_all.mean(), p_all.mean(), v_loss_np_all.mean(), v_p_all.mean(),
            ))


class TestModel(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=True, save_corres=False, save_w=True,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        self.save_w = save_w
        self.save_gt = save_gt
        checkpoint_path = EXP_path+'/EXP/{}/checkpoint/'.format(case_name)
        best_checkpoint = self.get_max_file_number(checkpoint_path)
        self.checkpoint_path_evl = EXP_path+'/EXP/{}/checkpoint/{}'.format(case_name, best_checkpoint)
        self.case_name = case_name
        '''set testing dataset'''
        print('set testing dataset')
        self.test_data, self.test_set = self.set_dataset(args.test_set, args)
        self.test_loader = DataLoader(self.test_data, 1, shuffle=False)
        self.gt_pairs = self.test_set['GT_pairs']

        '''set testing model'''
        print('set testing model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args

    def load_checkpoint_evl(self):
        checkpoint = torch.load(self.checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def calculate_area_opencv(self, points):
        # 将点转换为numpy数组
        contour = np.array(points)
        
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        
        return area
    
    def cosine_similarity(self, vec1, vec2):
        # Compute the dot product of two vectors
        dot_product = np.sum(vec1 * vec2, axis=1)
        # Calculate the L2 norm of each vector (i.e. the length of the vector)
        norm_vec1 = np.linalg.norm(vec1, axis=1)
        norm_vec2 = np.linalg.norm(vec2, axis=1)
        # Calculate cosine similarity
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def calculate_ratio(self, mixed_feature, vec_c, vec_t):
        # Calculate the sum of two vectors
        vec_contour = self.cosine_similarity(mixed_feature, vec_c)
        vec_texture = self.cosine_similarity(mixed_feature, vec_t)
        sum_vec = vec_contour + vec_texture
        ratio = np.divide(vec_contour, sum_vec, out=np.zeros_like(vec_contour), where=sum_vec!=0)
        return ratio
    

    def test_start(self):
        self.load_checkpoint_evl()
        gt_pairs = self.gt_pairs
        device = self.device
        global c
        valid_nums4 = 0  
        valid_nums2 = 0
        valid_nums6 = 0
        c = 0 
        w_min = 100
        w_count = 0
        haus_list = [] 

        '''test start'''
        print('test start!')
        saved_test_data = {
            "pred_transformation":[],
            "GT_transformation":[],
        }
        saved_test_weight = []
        for batch, (mask_para, imgs, pcd, c_input, t_input, adjs, factors, att_mask) in enumerate(tqdm(self.test_loader)):
            max_point_nums = len(pcd[0][0])
            adj_s = self.get_concat_adj(adjs[0], max_point_nums)
            adj_t = self.get_concat_adj(adjs[1], max_point_nums)

            source_input = {
                'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device), "att_mask":att_mask[0].to(device)
            }

            target_input = {
                'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device), "att_mask":att_mask[1].to(device)
            }

            pad_mask = self.get_pad_mask(mask_para).to(device)  # mark the padded part in similarity matrix
            mask = mask_para[0].to(device)
            feature_s, concat_source, w_s = self.models(source_input)
            feature_t, concat_target, w_t = self.models(target_input)
            similarity_matrix = self.get_similarity_matrix(feature_s, feature_t, pad_mask)

            w1 = w_s.clone().detach()
            w11 = w1.cpu().numpy()
            w2 = w_t.clone().detach()
            w22 = w2.cpu().numpy()
            saved_test_weight.append([w11,w22])

            '''visualization part'''
            gt_matrix = mask[0].to_dense().float().cpu().numpy()
            similarity_matrix = similarity_matrix[0].cpu().numpy()
            kernel = np.eye(3, dtype=np.uint8)
            kernel[1, 1] = 0
            kernel = np.rot90(kernel)
            similarity_matrix = cv2.erode(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            kernel[1, 1] = 1
            similarity_matrix = cv2.dilate(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

            idx_s, idx_t = gt_pairs[c]
            s_pcd_origin, t_pcd_origin = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]

            s_pcd, t_pcd = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]
            ind_s_origin, ind_t_origin = self.test_set['source_ind'][batch], self.test_set['target_ind'][batch]
            source_img, target_img = self.test_set['img_all'][idx_s], self.test_set['img_all'][idx_t]
            img_save_path = EXP_path+'/EXP/{}/result/img'.format(self.case_name)
            corres_save_path = EXP_path+'/EXP/{}/result/corres'.format(self.case_name)
            os.makedirs(img_save_path, exist_ok=True)
            os.makedirs(corres_save_path, exist_ok=True)
            evl = visualization.Visualization(gt_matrix, similarity_matrix, s_pcd, t_pcd, source_img,
                                              target_img, ind_s_origin, ind_t_origin, s_pcd_origin, t_pcd_origin, conv_threshold=0.006) # 0.006改成0.0006-》RANSAC很慢，改成0.06试试-》效果不好

            transformation, pairs = evl.get_transformation()



        
            if self.save_w:
                img_s = source_img.transpose(1, 0, 2)
                img_s = np.ascontiguousarray(img_s)
                evl.img_s = evl.weight_visualize(os.path.join(img_save_path, 'w_s{}.png'.format(c)),
                                                 img_s, s_pcd, w_s[0].detach().cpu().numpy())


                img_t = target_img.transpose(1, 0, 2)
                img_t = np.ascontiguousarray(img_t)
                evl.img_t = evl.weight_visualize(os.path.join(img_save_path, 'w_t{}.png'.format(c)),
                                                 img_t, t_pcd, w_t[0].detach().cpu().numpy())

            if self.save_img:
                evl.get_img(os.path.join(img_save_path, 'pred{}.png'.format(c)), transformation)

            # save ground truth result pairs
            if self.save_gt:
                evl.get_gt_img(os.path.join(img_save_path, 'gt{}.png'.format(c)))

            # save result pairs with corresponding points connected.
            if self.save_corres:
                evl.get_corresponding(os.path.join(corres_save_path, 'corres{}.png'.format(c)))

            # evaluation part
            if transformation is None:
                haus_list.append(0.)
                transformation = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0]])


            intersection_s = evl.pcd_s_inter_gt_origin
            intersection_s_trans = affine_transform(intersection_s, np.delete(transformation[:2], 2, axis=-1))
            intersection_t = evl.pcd_t_inter_gt_origin

            haus_dist = hausdorff_distance(intersection_s_trans, intersection_t, distance='euclidean')
            haus_list.append(haus_dist) 

            GT_transformation = evl.GT_transformation

            saved_test_data["pred_transformation"].append(np.delete(transformation[:2], 2, axis=-1))
            saved_test_data["GT_transformation"].append(GT_transformation)



            ermes = e_rmse(intersection_s_trans, intersection_t) 
            if ermes < 2:
                valid_nums2 += 1
            elif ermes < 4:
                valid_nums4 += 1
            elif ermes < 6:
                valid_nums6 += 1
            else:
                pass
            c += 1

        with open(EXP_path+'/EXP/{}/result/saved_test_exp_data.pkl'.format(self.case_name), 'wb') as file:
            pickle.dump(saved_test_data, file)

        print(w_count)
        registration_recall = (
        valid_nums2 / len(gt_pairs), valid_nums4 / len(gt_pairs), valid_nums6 / len(gt_pairs), float(w_min))
        with open(EXP_path+'/EXP/{}/result/registration recall.txt'.format(self.case_name), 'w') as f:
            f.write('{}'.format(registration_recall))

        with open(EXP_path+'/EXP/{}/result/haus_list.pkl'.format(self.case_name), 'wb') as f:
            pickle.dump(haus_list, f)


class STAGE_ONE(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=False, save_corres=False, save_w=False,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        self.save_w = save_w
        self.save_gt = save_gt
        checkpoint_path = EXP_path+'/EXP/{}/checkpoint/'.format(case_name)
        best_checkpoint = self.get_max_file_number(checkpoint_path)
        self.checkpoint_path_evl = EXP_path+'/EXP/{}/checkpoint/{}'.format(case_name, best_checkpoint)
        self.case_name = case_name
        if os.path.exists(EXP_path+'/EXP2/{}'.format(case_name)) is False:
            os.makedirs(EXP_path+'/EXP2/{}'.format(case_name))
        feature_save_path = args.stage2_feature_path+"/{}".format(case_name)
        self.saved_train_feature_path = feature_save_path+'/train_feature_{}.pkl'.format(args.dataset_select)
        self.saved_val_feature_path = feature_save_path+'/val_feature_{}.pkl'.format(args.dataset_select)
        self.saved_test_feature_path = feature_save_path+'/test_feature_{}.pkl'.format(args.dataset_select)
        if os.path.exists(feature_save_path) is False:
            os.makedirs(feature_save_path)
        
        '''set train dataset'''
        print('set training dataset')
        self.train_data, self.train_GT = self.set_dataset(args.train_set, args)
        self.train_loader = DataLoader(self.train_data, 1, num_workers=0,shuffle=False)
        self.train_gt_pairs = self.train_GT['GT_pairs']
        self.train_pcd = self.train_GT['full_pcd_all']
        self.s_index_train = self.train_GT['source_ind']
        self.t_index_train = self.train_GT['target_ind']

        '''set val dataset'''
        print('set testing dataset')
        self.valid_data, self.val_GT = self.set_dataset(args.valid_set, args)
        self.valid_loader = DataLoader(self.valid_data, 1, num_workers=0, shuffle=False)
        self.val_gt_pairs = self.val_GT['GT_pairs']
        self.val_pcd = self.val_GT['full_pcd_all']
        self.s_index_val = self.val_GT['source_ind']
        self.t_index_val = self.val_GT['target_ind']
        
        '''set test dataset'''
        print('set testing dataset')
        self.test_data, self.test_GT = self.set_dataset(args.test_set, args)
        self.test_loader = DataLoader(self.test_data, 1, num_workers=0, shuffle=False)
        self.test_gt_pairs = self.test_GT['GT_pairs']
        self.test_pcd = self.test_GT['full_pcd_all']
        self.s_index_test = self.test_GT['source_ind']
        self.t_index_test = self.test_GT['target_ind']

        '''set testing model'''
        print('set stage1 model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args

    #     self.saved_feature = {
    #     'train_feature': [],
    #     'val_feature': [],
    #     'test_feature': [],
    # }
    #    

    def load_checkpoint_evl(self):
        checkpoint = torch.load(self.checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file

    def stage1_start(self):
        self.load_checkpoint_evl()
        device = self.device
        '''save train feature'''
        train_saved_feature = {
            "saved_feature": [],
            "GT_pairs":self.train_gt_pairs,
            "full_pcd":self.train_pcd,
            "source_ind":self.s_index_train,
            "target_ind":self.t_index_train
        }
        for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.train_loader)):
            max_point_nums = len(pcd[0])
            # origin_adj = adj.clone()
            adj = self.get_concat_adj2(adj, max_point_nums)
            inputs = {
                'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
            }

            matching_feature, feature, _ = self.models(inputs) # bs,2611,64
            if self.args.stage2_data_model == "merged":
                train_saved_feature["saved_feature"].append(matching_feature.squeeze().cpu())
            else:
                train_saved_feature["saved_feature"].append(feature.squeeze().cpu())
            
            # train_saved_feature["adj"].append(origin_adj[0])
        
        with open(self.saved_train_feature_path, 'wb') as file:
            pickle.dump(train_saved_feature, file)

        '''save val feature'''
        val_saved_feature = {
            "saved_feature": [],
            "GT_pairs":self.val_gt_pairs,
            "full_pcd":self.val_pcd,
            "source_ind":self.s_index_val,
            "target_ind":self.t_index_val
        }
        for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.valid_loader)):
            max_point_nums = len(pcd[0])
            # origin_adj = adj.clone()
            adj = self.get_concat_adj2(adj, max_point_nums)
            inputs = {
                'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
            }

            matching_feature, feature, _ = self.models(inputs) # bs,2611,64
            if self.args.stage2_data_model == "merged":
                val_saved_feature["saved_feature"].append(matching_feature.squeeze().cpu())
            else:
                val_saved_feature["saved_feature"].append(feature.squeeze().cpu())
            # val_saved_feature["adj"].append(origin_adj[0])

        with open(self.saved_val_feature_path, 'wb') as file:
            pickle.dump(val_saved_feature, file)

        '''save test feature'''
        test_saved_feature = {
            "saved_feature": [],
            "GT_pairs":self.test_gt_pairs,
            "full_pcd":self.test_pcd,
            "source_ind":self.s_index_test,
            "target_ind":self.t_index_test
        }
        for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.test_loader)):
            max_point_nums = len(pcd[0])
            # origin_adj = adj.clone()
            adj = self.get_concat_adj2(adj, max_point_nums)
            inputs = {
                'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
            }

            matching_feature, feature, _ = self.models(inputs) # bs,2611,64

            if self.args.stage2_data_model == "merged":
                test_saved_feature["saved_feature"].append(matching_feature.squeeze().cpu())
            else:
                test_saved_feature["saved_feature"].append(feature.squeeze().cpu())
            # test_saved_feature["adj"].append(origin_adj[0])

        with open(self.saved_test_feature_path, 'wb') as file:
            pickle.dump(test_saved_feature, file)
        
        print("Stage 1 over")
        

class STAGE_TWO(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=False, save_corres=False, save_w=False,
                 save_gt=False):
        print('set training dataset')
        self.args = args
        self.saved_train_feature_path = args.stage2_feature_path+'/{}/train_feature_{}.pkl'.format(case_name, args.dataset_select)
        self.saved_val_feature_path = args.stage2_feature_path+'/{}/val_feature_{}.pkl'.format(case_name, args.dataset_select)
        self.saved_test_feature_path = args.stage2_feature_path+'/{}/test_feature_{}.pkl'.format(case_name, args.dataset_select)
        self.log_save_path = EXP_path+'/EXP2/{}/summary'.format(case_name)
        self.writer = SummaryWriter(self.log_save_path)
        self.case_name = case_name
        self.checkpoint_path = EXP_path+'/EXP2/{}/checkpoint'.format(case_name)
        self.max_point = args.max_length
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)

        # DDP
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        random_seed = 20
        init_seeds(random_seed+torch.distributed.get_rank())
        # local_rank = args.local_rank
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        models = net(args)
        models = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models).cuda() 
        self.models = DDP(models, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)



        self.train_data, _ = self.set_dataset_searching(self.saved_train_feature_path, args)
        self.train_sampler = DistributedSampler(self.train_data)
        self.train_loader = DataLoader(self.train_data, args.batch_size, num_workers=0, sampler=self.train_sampler)
        # self.train_loader = DataLoader(self.train_data, args.batch_size, num_workers=0, shuffle=True)

        self.val_data, _ = self.set_dataset_searching(self.saved_val_feature_path, args)
        self.val_loader = DataLoader(self.val_data, args.batch_size, num_workers=0, sampler=DistributedSampler(self.val_data))
        # self.val_loader = DataLoader(self.val_data, args.batch_size, num_workers=0, shuffle=True)
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # self.models.to(self.device)
        self.contrast_temperature = args.contrast_temperature
        self.epoch = args.stage2_epoch

        self.optimizer = torch.optim.Adam(self.models.parameters(),
                                          lr=args.stage2_lr,
                                          weight_decay=args.stage2_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.stage2_epoch)

    def warmup(self, current_step: int):
        return 1 / (10 ** (float(self.args.warmup_steps - current_step)))
    
    def set_dataset_searching(self, data_path, args):
        with open(data_path, 'rb') as feature_file:
            stage1_features = pickle.load(feature_file)
        # gt_config['model_type'] = args.model_type
        # gt_config['channel'] = args.channel
        # gt_config['c_model'] = args.c_model
        # gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyDataSet_searching(stage1_features, args)
        return dataset, None
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + '/stageTwo_model_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({  # 'state': torch.cuda.get_rng_state_all(),
                'epoch': epoch,
                'model_state_dict': self.models.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, path)
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    def calculate_train_val_top_recall(self, batch_feature_s, batch_feature_t):
        F_normalized_s = F.normalize(batch_feature_s, p=2, dim=1)
        F_normalized_t = F.normalize(batch_feature_t, p=2, dim=1)
        bs = batch_feature_s.shape[0]
        GT_pairs = []
        for i in range(0, bs):
            GT_pairs.append([i,i])
        cos_sim_matrix = torch.matmul(F_normalized_s, F_normalized_t.T)
        sort_matrix = torch.sort(cos_sim_matrix, dim=-1, descending=True)

        idx = sort_matrix[1]
        idx = idx.numpy() #（3279，3279）  gt_pair（2370，2）
        l = []
        for i in range(len(GT_pairs)):
            # if mins < len_all[i] <= maxs:
            l.append(np.argwhere(idx[GT_pairs[i][0]] == GT_pairs[i][1]))

        result = np.array(l).reshape(-1)
        top1 = (result < 1).sum() / len(l)
        top5 = (result < 5).sum() / len(l)
        top10 = (result < 10).sum() / len(l)
        top20 = (result < 20).sum() / len(l)

        return torch.tensor([top1]), torch.tensor([top5])
    
    def get_mask(self, logits, s_index, t_index):
        # s_index = GT_pairs[3]
        # t_index = GT_pairs[4]
        s_list = self.index_tensor(s_index)
        t_list =  self.index_tensor(t_index)

        mask = torch.eye(logits.shape[0], logits.shape[1], dtype=bool).to(logits.device)
        mask[s_list, t_list] = True

        return mask

    def index_tensor(self, tnsr):
        #一个batch后面如果重复出现同一个样本，使用第一次出现的位置的索引
        result = []
        for i in range(tnsr.shape[0]):
            if tnsr[i] in tnsr[:i]:
                # a = (tnsr == tnsr[i]).nonzero(as_tuple=True)[0][0]
                result.append((tnsr == tnsr[i]).nonzero(as_tuple=True)[0][0].item())
            else:
                result.append(i)
        return torch.tensor(result)
    
    def get_similarity_matrix(self, feature1, feature2, pad_mask):
        similarity_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1)) / self.temperature
        similarity_matrix[pad_mask] -= 1e9  # give a very small value to the padded part for softmax operation
        s_i = torch.softmax(similarity_matrix, dim=1)  # row softmax
        s_j = torch.softmax(similarity_matrix, dim=-1)  # column softmax
        similarity_matrix = torch.multiply(s_i, s_j)

        return similarity_matrix
    
    def train_start(self):
        
        print('Stage 2 training start!!!')
        infor_loss_train = torch.zeros([0])
        infor_loss_val = torch.zeros([0])

        top1_reacll_train = torch.zeros([0])
        top5_reacll_train = torch.zeros([0])
        top1_reacll_val = torch.zeros([0])
        top5_reacll_val = torch.zeros([0])

        loss_m_all = torch.zeros([0])
        v_loss_np_all = torch.zeros([0])
        p_all = torch.zeros([0])
        v_p_all = torch.zeros([0])

        min_loss = torch.inf
        for i in range(self.epoch):

            self.models.train()
            self.models.requires_grad_(True)

            for e, (stage1_features) in enumerate(tqdm(self.train_loader)):
                self.train_sampler.set_epoch(e)
                all_data, mask_para = stage1_features
                stage1_features_s, stage1_features_t, index_s, index_t, pcd_s, pcd_t = all_data
                stage1_features_s, stage1_features_t, pcd_s, pcd_t = stage1_features_s.to(self.device), stage1_features_t.to(self.device), pcd_s.to(self.device), pcd_t.to(self.device)


                feature_s, _ = self.models(stage1_features_s, pcd_s)
                feature_t, _ = self.models(stage1_features_t, pcd_t)

                
                '''searching loss'''
                InfoNCE_loss = InfoNCE(temperature=self.contrast_temperature)
                infor_loss = InfoNCE_loss(feature_s, feature_t, gt_pairs=(index_s, index_t))

                infor_loss_s = InfoNCE_loss(feature_s, feature_s, gt_pairs=(index_s, index_s))
                infor_loss_t = InfoNCE_loss(feature_t, feature_t, gt_pairs=(index_t, index_t))

                only_negative_weight = 0.5
                total_loss = infor_loss + (only_negative_weight*infor_loss_s + only_negative_weight*infor_loss_t)/2



                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                infor_loss_train = torch.cat((infor_loss_train, infor_loss.detach().cpu().view(-1)))
                # loss_m_all = torch.cat((loss_m_all, loss_np.detach().cpu().view(-1)))
                # p_all = torch.cat((p_all, loss_p.cpu().view(-1)))
                loss_m_all = torch.cat((loss_m_all, torch.zeros([0])))
                p_all = torch.cat((p_all, torch.zeros([0])))

                import ipdb
                # ipdb.set_trace()
                top1_train, top5_train = self.calculate_train_val_top_recall(feature_s.detach().cpu(), feature_t.detach().cpu())
                top1_reacll_train = torch.cat((top1_reacll_train, top1_train))
                top5_reacll_train = torch.cat((top5_reacll_train, top5_train))

            self.scheduler.step()

            if torch.distributed.get_rank() in [1]:
                self.writer.add_scalar('train_loss', loss_m_all.mean(), i)
                self.writer.add_scalar('Stage2_Infor_loss_train', infor_loss_train.mean(), i)

                self.writer.add_scalar('top1_reacll_train', top1_reacll_train.mean(), i)
                self.writer.add_scalar('top5_reacll_train', top5_reacll_train.mean(), i)


            if (i + 1) % 2 != 0:
                continue
            '''validation'''
            self.models.eval()
            self.models.requires_grad_(False)
            for batch, (stage1_features) in enumerate(tqdm(self.val_loader)):
                all_data, mask_para = stage1_features
                stage1_features_s, stage1_features_t, index_s, index_t, pcd_s, pcd_t = all_data
                stage1_features_s, stage1_features_t, pcd_s, pcd_t = stage1_features_s.to(self.device), stage1_features_t.to(self.device), pcd_s.to(self.device), pcd_t.to(self.device)
                


                feature_s, _ = self.models(stage1_features_s, pcd_s)
                feature_t, _ = self.models(stage1_features_t, pcd_t)

                '''searching loss'''
                InfoNCE_loss = InfoNCE(temperature=self.contrast_temperature)
                infor_loss = InfoNCE_loss(feature_s, feature_t, gt_pairs=(index_s, index_t))

                v_loss_np_all = torch.cat((v_loss_np_all, torch.zeros([0])))
                v_p_all = torch.cat((v_p_all,torch.zeros([0])))
                

                infor_loss_val = torch.cat((infor_loss_val, infor_loss.detach().cpu().view(-1)))

                top1_val, top5_val = self.calculate_train_val_top_recall(feature_s.detach().cpu(), feature_t.detach().cpu())
                top1_reacll_val = torch.cat((top1_reacll_val, top1_val))
                top5_reacll_val = torch.cat((top5_reacll_val, top5_val))
            
            
            if torch.distributed.get_rank() in [1]:

                self.writer.add_scalar('top1_reacll_val', top1_reacll_val.mean(), i)
                self.writer.add_scalar('top5_reacll_val', top5_reacll_val.mean(), i)

                self.writer.add_scalar('StageTwo_infor_loss_val', infor_loss_val.mean(), i)
                self.writer.add_scalar('valid_loss', v_loss_np_all.mean(), i)
                self.writer.add_scalar('valid_positive_loss', v_p_all.mean(), i)
                means_all = infor_loss_val.mean()

                if means_all < min_loss:
                    self.save_checkpoint(i)

                print("loacl rank = {}".format(0))
                print("epoch = {}".format(i))
                print('epoch = {}, match_loss = {}, loss_p = {}, v_match_loss = {}, v_loss_p = {}'.format(
                    i, loss_m_all.mean(), p_all.mean(), v_loss_np_all.mean(), v_p_all.mean(),
                ))
                print('\n')
                print("stage2_infor_loss_val =  {}".format(infor_loss_val.mean()))
                print("top1_reacll_val =  {}".format(top1_reacll_val.mean()))
                print("top5_reacll_val =  {}".format(top5_reacll_val.mean()))


class ST2_SearchModel(object):
    def __init__(self, net, args, temperature, case_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.case_name = case_name
        self.checkpoint_path = EXP_path+'/EXP2/{}/checkpoint'.format(case_name)
        self.saved_test_feature_path = args.stage2_feature_path+'/{}/test_feature_{}.pkl'.format(case_name, args.dataset_select)
        self.test_data, self.Stage1_data = self.set_dataset_searching(self.saved_test_feature_path, args)
        self.test_loader = DataLoader(self.test_data, 1, num_workers=0, shuffle=False)
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.feature_all_flatten = torch.zeros((0, args.global_out_channels))
        self.max_point = args.max_length
        self.global_out_channels = args.global_out_channels


    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def set_dataset_searching(self, data_path, args):
        with open(data_path, 'rb') as feature_file:
            stage1_features = pickle.load(feature_file)
        dataset = data_preprocess.MyDataSet_searching(stage1_features, args)
        return dataset, stage1_features

    def load_checkpoint_evl(self, checkpoint_path_evl):
        checkpoint = torch.load(checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return

    def feature_searching(self, result_matrix, gt_pair):
        """to get the topk searching result from score matrix"""
        sort_matrix = torch.sort(result_matrix, dim=-1, descending=True)
        idx = sort_matrix[1]
        idx = idx.numpy() 
        l = []
        for i in range(len(gt_pair)):
            l.append(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))

        result = np.array(l).reshape(-1)

        top1 = (result < 1).sum() / len(l)
        top5 = (result < 5).sum() / len(l)
        top10 = (result < 10).sum() / len(l)
        top20 = (result < 20).sum() / len(l)

        return top1, top5, top10, top20
    
    @staticmethod
    def get_concat_adj2(adj, max_len):
        device = adj.device
        temp_adj = torch.zeros((2, 0), dtype=torch.int).to(device)
        for i in range(len(adj)):
            b = torch.nonzero(adj[i]).transpose(0, 1)
            temp_adj = torch.hstack((temp_adj, b + i * max_len))

        return temp_adj
    
    
    def searching_start(self):
        best_checkpoint = self.get_max_file_number(self.checkpoint_path)
        print("best_checkpoint:{}".format(best_checkpoint))

        checkpoint_path_evl = EXP_path+'/EXP2/{}/checkpoint/{}'.format(self.case_name, best_checkpoint)
        self.load_checkpoint_evl(checkpoint_path_evl)

        self.stage2_result_path = EXP_path+'/EXP2/{}/result'.format(self.case_name)
        if os.path.exists(self.stage2_result_path) is False:
            os.mkdir(self.stage2_result_path)
        
        import time
        start_time = time.time()
        saved_test_weight = []
        for batch, (stage1_features) in enumerate(tqdm(self.test_loader)):
            stage1_features, pcd = stage1_features
            stage1_features, pcd = stage1_features.to(self.device), pcd.to(self.device)
            
            F_global, w_ = self.models(stage1_features, pcd)

            self.feature_all_flatten = torch.cat((self.feature_all_flatten, F_global.cpu()), dim=0)

            w = w_.clone().detach()
            w = w.cpu().numpy()
            saved_test_weight.append(w)

        F_normalized = F.normalize(self.feature_all_flatten, p=2, dim=1)

        cos_sim_matrix = torch.matmul(F_normalized, F_normalized.T)
        cos_sim_matrix.fill_diagonal_(-1)
        result = self.feature_searching(cos_sim_matrix, self.Stage1_data['GT_pairs'])
        print(result)
        end_time = time.time()
        run_time = (end_time - start_time)
        print("Runtime：", run_time, "s")



        saved_matrix = {
            "matrix": cos_sim_matrix.data.cpu().numpy(),
            "GT_pairs": self.Stage1_data['GT_pairs']
        }


        with open(self.stage2_result_path+'/sim_matrix_390_stage2_self_gate.pkl', 'wb') as f:
            pickle.dump(saved_matrix, f)
        with open(self.stage2_result_path+'/global_feature_390_stage2_self_gate.pkl', 'wb') as f:
            pickle.dump(saved_matrix, f)
        searching_recall = result
        with open(self.stage2_result_path+'/searching recall.txt', 'w') as f:
            f.write('{}'.format(searching_recall))
        
        with open(self.stage2_result_path+'/saved_test_weight_390_stage2.pkl'.format(self.case_name), 'wb') as file:
            pickle.dump(saved_test_weight, file)



    def searching_start_every_model(self):

        max_value = [0]

        for file in os.listdir(self.checkpoint_path):
            best_checkpoint = file
            self.feature_all_flatten = torch.zeros((0, self.global_out_channels))

            print("checkpoint:{}".format(best_checkpoint))

            checkpoint_path_evl = EXP_path+'/EXP2/{}/checkpoint/{}'.format(self.case_name, best_checkpoint).format(self.case_name, best_checkpoint)
            self.load_checkpoint_evl(checkpoint_path_evl)

            for batch, (stage1_features) in enumerate(tqdm(self.test_loader)):
                stage1_features, pcd = stage1_features
                stage1_features, pcd = stage1_features.to(self.device), pcd.to(self.device)


                F_global, _ = self.models(stage1_features, pcd)

                self.feature_all_flatten = torch.cat((self.feature_all_flatten, F_global.cpu()), dim=0)

            F_normalized = F.normalize(self.feature_all_flatten, p=2, dim=1)
            cos_sim_matrix = torch.matmul(F_normalized, F_normalized.T)
            cos_sim_matrix.fill_diagonal_(-1)
            result = self.feature_searching(cos_sim_matrix, self.Stage1_data['GT_pairs'])
            print(result)
            if result[0] > max_value[0]:
                max_value = result
        print("best test checkpoint:")
        print(result)


class Real_TestModel(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=True, save_corres=False, save_w=False,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        self.save_w = save_w
        self.save_gt = save_gt
        checkpoint_path = EXP_path+'/EXP/{}/checkpoint/'.format(case_name)
        best_checkpoint = self.get_max_file_number(checkpoint_path)
        self.checkpoint_path_evl = EXP_path+'/EXP/{}/checkpoint/{}'.format(case_name, best_checkpoint)
        self.case_name = case_name
        '''set testing dataset'''
        print('set testing dataset')
        self.test_data, self.test_set = self.set_real_dataset(args.real_test_set, args)
        self.test_loader = DataLoader(self.test_data, 1, shuffle=False)
        self.gt_pairs = self.test_set['GT_pairs']

        '''set testing model'''
        print('set testing model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args
    
    def set_real_dataset(self, data_path, args):
        with open(data_path, 'rb') as gt_file:
            gt_config = pickle.load(gt_file)

        gt_config['model_type'] = args.model_type
        gt_config['channel'] = args.channel
        gt_config['c_model'] = args.c_model
        gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyRealDataSet(gt_config, args)
        return dataset, gt_config
    
    def get_similarity_matrix_real_test(self, feature1, feature2, pad_mask):
        similarity_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1)) / self.temperature
        similarity_matrix[pad_mask] -= 1e9  # give a very small value to the padded part for softmax operation
        s_i = torch.softmax(similarity_matrix, dim=1)  # row softmax
        s_j = torch.softmax(similarity_matrix, dim=-1)  # column softmax
        similarity_matrix = torch.multiply(s_i, s_j)

        return similarity_matrix
    

    def load_checkpoint_evl(self):
        checkpoint = torch.load(self.checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def calculate_area_opencv(self, points):
        # 将点转换为numpy数组
        contour = np.array(points)
        
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        
        return area
    
    def cosine_similarity(self, vec1, vec2):
        # 计算两个向量的点积
        dot_product = np.sum(vec1 * vec2, axis=1)
        # 计算每个向量的L2范数（即向量的长度）
        norm_vec1 = np.linalg.norm(vec1, axis=1)
        norm_vec2 = np.linalg.norm(vec2, axis=1)
        # 计算余弦相似度
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def calculate_ratio(self, mixed_feature, vec_c, vec_t):
        # 计算两个向量的和
        vec_contour = self.cosine_similarity(mixed_feature, vec_c)
        vec_texture = self.cosine_similarity(mixed_feature, vec_t)
        sum_vec = vec_contour + vec_texture
        # 计算第一个向量的每一个位置元素除以两个向量该位置的元素的和
        ratio = np.divide(vec_contour, sum_vec, out=np.zeros_like(vec_contour), where=sum_vec!=0)
        return ratio
    

    def test_start(self):
        self.load_checkpoint_evl()
        gt_pairs = self.gt_pairs
        device = self.device
        global c
        valid_nums4 = 0  # count the good registration nums
        valid_nums2 = 0
        valid_nums6 = 0
        c = 0  # count the fragment nums
        w_min = 100
        w_count = 0
        haus_list = []  # list of mean hausdroff distance of each gt pair

        '''test start'''
        print('test start!')
        saved_test_data = {
            "pred_real_transformation":[],
            # "GT_transformation":[],
        }
        saved_test_weight = []
        for batch, (mask_para, imgs, pcd, c_input, t_input, adjs, factors) in enumerate(tqdm(self.test_loader)):

            max_point_nums = len(pcd[0][0])
            adj_s = self.get_concat_adj(adjs[0], max_point_nums)
            adj_t = self.get_concat_adj(adjs[1], max_point_nums)

            source_input = {
                'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device)
            }

            target_input = {
                'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device)
            }

            pad_mask = self.get_pad_mask(mask_para).to(device)  # mark the padded part in similarity matrix
            # mask = mask_para[0].to(device)
            feature_s, concat_source, w_s = self.models(source_input)
            feature_t, concat_target, w_t = self.models(target_input)
            similarity_matrix = self.get_similarity_matrix_real_test(feature_s, feature_t, pad_mask)


            '''visualization part'''
            similarity_matrix = similarity_matrix[0].cpu().numpy()
            kernel = np.eye(3, dtype=np.uint8)
            kernel[1, 1] = 0
            kernel = np.rot90(kernel)
            similarity_matrix = cv2.erode(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            kernel[1, 1] = 1
            similarity_matrix = cv2.dilate(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

            idx_s, idx_t = gt_pairs[batch]
            s_pcd_origin, t_pcd_origin = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]
            # s_pcd, t_pcd = self.test_set['down_sample_pcd'][idx_s], self.test_set['down_sample_pcd'][idx_t]
            s_pcd, t_pcd = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]
            # ind_s_origin, ind_t_origin = self.test_set['source_ind'][batch], self.test_set['target_ind'][batch]
            source_img, target_img = self.test_set['img_all'][idx_s], self.test_set['img_all'][idx_t]
            img_save_path = EXP_path+'/EXP/{}/result/real_img'.format(self.case_name)
            corres_save_path = EXP_path+'/EXP/{}/result/corres'.format(self.case_name)
            os.makedirs(img_save_path, exist_ok=True)
            os.makedirs(corres_save_path, exist_ok=True)
            evl = visualization.Visualization_real_data(similarity_matrix, s_pcd, t_pcd, source_img,
                                              target_img, s_pcd_origin, t_pcd_origin, conv_threshold=0.006) 

            transformation, pairs = evl.get_transformation()


            # get weighted img
            # w_s = 1- w_s
            if self.save_w:
                img_s = source_img.transpose(1, 0, 2)
                img_s = np.ascontiguousarray(img_s)
                evl.img_s = evl.weight_visualize(os.path.join(img_save_path, 'w_s{}.png'.format(batch)),
                                                 img_s, s_pcd, w_s[0].detach().cpu().numpy())


                img_t = target_img.transpose(1, 0, 2)
                img_t = np.ascontiguousarray(img_t)
                evl.img_t = evl.weight_visualize(os.path.join(img_save_path, 'w_t{}.png'.format(batch)),
                                                 img_t, t_pcd, w_t[0].detach().cpu().numpy())

            # save predicted result pairs
            if self.save_img:
                evl.get_img(os.path.join(img_save_path, 'pred{}.png'.format(batch)), transformation)

        with open(EXP_path+'/EXP/{}/result/saved_real_test_data.pkl'.format(self.case_name), 'wb') as file:
            pickle.dump(saved_test_data, file)



class STAGE_ONE_REAL(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=False, save_corres=False, save_w=False,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        # self.save_w = save_w
        # self.save_gt = save_gt
        checkpoint_path = EXP_path+'/EXP/{}/checkpoint/'.format(case_name)
        best_checkpoint = self.get_max_file_number(checkpoint_path)
        self.checkpoint_path_evl = EXP_path+'/EXP/{}/checkpoint/{}'.format(case_name, best_checkpoint)
        self.case_name = case_name
        if os.path.exists(EXP_path+'/EXP2/{}'.format(case_name)) is False:
            os.makedirs(EXP_path+'/EXP2/{}'.format(case_name))
        feature_save_path = args.real_stage2_feature_path+"/{}".format(case_name)
        self.saved_train_feature_path = feature_save_path+'/real_img_feature_{}.pkl'.format(args.dataset_select)

        if os.path.exists(feature_save_path) is False:
            os.makedirs(feature_save_path)
        
        '''set train dataset'''
        print('set training dataset')
        self.train_data, self.train_GT = self.set_real_dataset(args.real_test_set, args)
        self.train_loader = DataLoader(self.train_data, 1, num_workers=0,shuffle=False)
        self.train_gt_pairs = self.train_GT['GT_pairs']
        self.train_pcd = self.train_GT['full_pcd_all']


        '''set testing model'''
        print('set stage1 model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args

    def set_real_dataset(self, data_path, args):
        with open(data_path, 'rb') as gt_file:
            gt_config = pickle.load(gt_file)

        gt_config['model_type'] = args.model_type
        gt_config['channel'] = args.channel
        gt_config['c_model'] = args.c_model
        gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyRealDataSet(gt_config, args)
        return dataset, gt_config


    def load_checkpoint_evl(self):
        checkpoint = torch.load(self.checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file

    def stage1_start(self):
        self.load_checkpoint_evl()
        device = self.device
        '''save train feature'''
        train_saved_feature = {
            "saved_feature": [],
            "GT_pairs":self.train_gt_pairs,
            "full_pcd":self.train_pcd,
        }
        for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.train_loader)):
            max_point_nums = len(pcd[0])

            adj = self.get_concat_adj2(adj, max_point_nums)
            inputs = {
                'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
            }

            matching_feature, feature, _ = self.models(inputs) # bs,2611,64
            train_saved_feature["saved_feature"].append(feature.squeeze().cpu())
            
        
        with open(self.saved_train_feature_path, 'wb') as file:
            pickle.dump(train_saved_feature, file)

        
        print("Stage 1 Real over")

class ST2_Real_SearchModel(object):
    def __init__(self, net, args, temperature, case_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.case_name = case_name
        self.checkpoint_path = EXP_path+'/EXP2/{}/checkpoint'.format(case_name)
        self.saved_test_feature_path = args.real_stage2_feature_path+'/{}/real_img_feature_{}.pkl'.format(case_name, args.dataset_select)
        self.test_data, self.Stage1_data = self.set_dataset_searching(self.saved_test_feature_path, args)
        self.test_loader = DataLoader(self.test_data, 1, num_workers=0, shuffle=False)
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.feature_all_flatten = torch.zeros((0, args.global_out_channels))
        self.max_point = args.max_length


    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def set_dataset_searching(self, data_path, args):
        with open(data_path, 'rb') as feature_file:
            stage1_features = pickle.load(feature_file)
        dataset = data_preprocess.MyRealDataSet_searching(stage1_features, args)
        return dataset, stage1_features

    def load_checkpoint_evl(self, checkpoint_path_evl):
        checkpoint = torch.load(checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return

    def feature_searching(self, result_matrix, gt_pair):
        """to get the topk searching result from score matrix"""
        # result_matrix = result_matrix + result_matrix.T
        sort_matrix = torch.sort(result_matrix, dim=-1, descending=True)
        # sort_matrix = torch.sort(result_matrix, dim=-1, descending=False)
        idx = sort_matrix[1]
        idx = idx.numpy() #（3279，3279）  gt_pair（2370，2）
        l = []
        bad_list = []
        for i in range(len(gt_pair)):
            # if mins < len_all[i] <= maxs:
            l.append(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))
            a = int(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))
            if a > 20:
                bad_list.append(i)

        result = np.array(l).reshape(-1)

        top1 = (result < 1).sum() / len(l)
        top5 = (result < 5).sum() / len(l)
        top10 = (result < 10).sum() / len(l)
        top20 = (result < 20).sum() / len(l)

        return top1, top5, top10, top20
    
    @staticmethod
    def get_concat_adj2(adj, max_len):
        device = adj.device
        temp_adj = torch.zeros((2, 0), dtype=torch.int).to(device)
        for i in range(len(adj)):
            b = torch.nonzero(adj[i]).transpose(0, 1)
            # a = adj[i].coalesce().indices() #(2,8602)
            temp_adj = torch.hstack((temp_adj, b + i * max_len))

        return temp_adj
    
    
    def searching_start(self):
        best_checkpoint = self.get_max_file_number(self.checkpoint_path)
        print("best_checkpoint:{}".format(best_checkpoint))

        checkpoint_path_evl = EXP_path+'/EXP2/{}/checkpoint/{}'.format(self.case_name, best_checkpoint).format(self.case_name, best_checkpoint)
        self.load_checkpoint_evl(checkpoint_path_evl)

        for batch, (stage1_features) in enumerate(tqdm(self.test_loader)):
            stage1_features, pcd = stage1_features
            stage1_features, pcd = stage1_features.to(self.device), pcd.to(self.device)

            max_point_nums = self.max_point
            # adj = self.get_concat_adj2(adj, max_point_nums)
            
            F_global, _ = self.models(stage1_features, pcd)

            self.feature_all_flatten = torch.cat((self.feature_all_flatten, F_global.cpu()), dim=0)

        F_normalized = F.normalize(self.feature_all_flatten, p=2, dim=1)
        # 计算余弦相似度矩阵
        cos_sim_matrix = torch.matmul(F_normalized, F_normalized.T)
        cos_sim_matrix.fill_diagonal_(-1)
        result = self.feature_searching(cos_sim_matrix, self.Stage1_data['GT_pairs'])
        print(result)

        calute_NDCG(cos_sim_matrix, self.Stage1_data['GT_pairs'])


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    opt = config.args

    '''set 温度系数'''
    temp = math.sqrt(opt.feature_dim)

    net = pipeline.Vanilla
    ST2_net = pipeline.TransformerEncoderModel

    exp_name = 'exp1' 

    EXP_path = opt.exp_path
    if opt.model_type == 'matching_train': 
        trainer = Train_model(net, opt, temp, exp_name)
        trainer.train_start()
    elif opt.model_type == 'matching_test': 
        tester = TestModel(net, opt, temp, exp_name)
        tester.test_start()
    elif opt.model_type == 'save_stage1_feature': 
        ST1 = STAGE_ONE(net, opt, temp, exp_name)
        ST1.stage1_start()
    elif opt.model_type == 'searching_train':
        ST2 = STAGE_TWO(ST2_net, opt, temp, exp_name)
        ST2.train_start()
    elif opt.model_type == 'searching_test':
        ST2_searcher = ST2_SearchModel(ST2_net, opt, temp, exp_name)
        ST2_searcher.searching_start()

    elif opt.model_type == 'real_dataset_test':
        Real_tester = Real_TestModel(net, opt, temp, exp_name)
        Real_tester.test_start()
    elif opt.model_type == 'stage1_real':  # --------------------stage1:save feature---------------------
        ST1 = STAGE_ONE_REAL(net, opt, temp, exp_name)
        ST1.stage1_start()
    elif opt.model_type == 'stage2_real_searching':
        ST2_searcher = ST2_Real_SearchModel(ST2_net, opt, temp, exp_name)
        ST2_searcher.searching_start()
