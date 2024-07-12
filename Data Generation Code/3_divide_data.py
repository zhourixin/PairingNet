import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


# shujv huafen

def judge_where(dot1, dot2, cur):
    if cur < dot1:
        return 1
    elif cur >= dot2:
        return 3
    else:
        return 2


def divide(data_path):
    train_set = {
        'full_pcd_all': [],
        'img_all': [],
        'extra_img': [],
        'shape_all': [],
        'GT_pairs': [],
        'source_ind': [],
        'target_ind': [],
        'intersection_len': [],
    }
    valid_set = {
        'full_pcd_all': [],
        'img_all': [],
        'extra_img': [],
        'shape_all': [],
        'GT_pairs': [],
        'source_ind': [],
        'target_ind': [],
        'intersection_len': [],
    }
    test_set = {
        'full_pcd_all': [],
        'img_all': [],
        'extra_img': [],
        'shape_all': [],
        'GT_pairs': [],
        'source_ind': [],
        'target_ind': [],
        'intersection_len': [],
    }

    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    nums = len(data['img_all'])
    shuffle_ind = np.arange(0, nums)
    cur_ind = np.arange(0, nums)
    random.shuffle(shuffle_ind)
    shuffle_cur_dic = {shuffle_ind[i]: cur_ind[i] for i in range(nums)}
    dot1, dot2 = (nums * 5)//10, (nums * 6)//10
    for key in data:
        if key in ['img_all', 'extra_img', 'full_pcd_all', 'shape_all']:
            train_set[key] = [data[key][idx] for idx in shuffle_ind[:dot1]]
            valid_set[key] = [data[key][idx] for idx in shuffle_ind[dot1:dot2]]
            test_set[key] = [data[key][idx] for idx in shuffle_ind[dot2:]]

    for i, pair in enumerate(data['GT_pairs']):
        cur1, cur2 = shuffle_cur_dic.get(pair[0]), shuffle_cur_dic.get(pair[1])
        where1, where2 = judge_where(dot1, dot2, cur1), judge_where(dot1, dot2, cur2)
        if where1 == where2: 
            if where1 == 1:
                train_set['GT_pairs'].append([cur1, cur2])
                for key in train_set:
                    if key in ['source_ind', 'target_ind']:
                        train_set[key].append(data[key][i])

            elif where1 == 2:
                valid_set['GT_pairs'].append([cur1-dot1, cur2-dot1])
                for key in valid_set:
                    if key in ['source_ind', 'target_ind']:
                        valid_set[key].append(data[key][i])
            else:
                test_set['GT_pairs'].append([cur1-dot2, cur2-dot2])
                for key in test_set:
                    if key in ['source_ind', 'target_ind']:
                        test_set[key].append(data[key][i])
    print('stop')
    with open(R_PATH+'/ori_train_set.pkl', 'wb') as file:
        pickle.dump(train_set, file)
    with open(R_PATH+'/ori_valid_set.pkl', 'wb') as file:
        pickle.dump(valid_set, file)
    with open(R_PATH+'/ori_test_set.pkl', 'wb') as file:
        pickle.dump(test_set, file)


if __name__ == '__main__':
    R_PATH = "./"
    root = R_PATH+'/ori_dataset_all.pkl'
    divide(root)
