import random
import collections
import numpy as np
from sklearn.neighbors import KDTree
from utilz import rigid_transform_2d
import math

def iter_match(
    pcd_source, pcd_target,
    proposal,
    checker_params
):
    proposal.sort()
    source_pcd, target_pcd = pcd_source[proposal], pcd_target[proposal]

    # c. fast correspondence distance check:s
    source_pcd = np.pad(source_pcd, ((0, 0), (0, 1)), 'constant', constant_values=1)
    target_pcd = np.pad(target_pcd, ((0, 0), (0, 1)), 'constant', constant_values=1)
    T = rigid_transform_2d(source_pcd, target_pcd)  # 通过 svd 初步求解 旋转、平移矩阵
    T = np.vstack((T, np.array([0., 0., 0., 1.], dtype=np.float32)))
    R, t = T[0:3, 0:3], T[0:3, 3]

    # deviation：偏差  区分 inline outline 通过 距离判断
    deviation = np.linalg.norm(
       target_pcd - np.dot(source_pcd, R.T) - t,
       axis=1
    )

    is_valid_correspondence_distance = np.all(deviation <= checker_params.max_correspondence_distance)

    return T, source_pcd, target_pcd, deviation, is_valid_correspondence_distance
    # if is_valid_correspondence_distance else None


def ransac_match(source_pair, target_pair):
    distance_threshold_init = 10
    # RANSAC configuration:
    ransac_para = collections.namedtuple(
        'RANSACParams',
        [
            'max_workers',
            'num_samples',
            'max_correspondence_distance', 'max_iter', 'max_validation',
            'd'
        ]
    )
    # fast pruning algorithm configuration:
    check_para = collections.namedtuple(
        'CheckerParams',
        ['max_correspondence_distance', 'max_edge_length_ratio', 'normal_angle_threshold']
    )
    ransac_para = ransac_para(
        max_workers=8,
        num_samples=5,# 测试了4，5，6，8，还是5效果好，更深一层原因有待考证
        max_correspondence_distance=distance_threshold_init,
        max_iter=2000,
        max_validation=10,
        d=15
    )


    check_para = check_para(
        max_correspondence_distance=distance_threshold_init,
        max_edge_length_ratio=0.9,
        normal_angle_threshold=None
    )

    n = len(source_pair)
    idx_matches = np.arange(n)

    validator = lambda proposal: iter_match(source_pair, target_pair, proposal[0], check_para)

    max_fitness = -1
    best_result = None
    best_pair = None

    max_fitness_less = -1
    best_result_less = None
    best_pair_less = None
    # ransac:
    num_validation = 0
    for i in range(ransac_para.max_iter):
        # proposal = np.random.choice(idx_matches, ransac_para.num_samples, replace=False)
        proposal = np.arange(0, len(idx_matches))
        random.shuffle(proposal)
        proposal = proposal[:ransac_para.num_samples]
        T = validator((proposal, i))
        # if (not (T is None)) and (num_validation < ransac_para.max_validation):
        transformation = np.delete(T[0][:2], 2, axis=-1)
        changed_pair_s = np.matmul(np.hstack((source_pair, np.ones((len(source_pair), 1)))), transformation.T)
        deviation = np.linalg.norm(
            target_pair - changed_pair_s,
            axis=1
        )
        if T[4] == True and (num_validation < ransac_para.max_validation) and np.sum(deviation<10)>ransac_para.d:
            num_validation += 1
            # update best result:
            transformation = np.delete(T[0][:2], 2, axis=-1)
            changed_pair_s = np.matmul(np.hstack((source_pair, np.ones((len(source_pair), 1)))), transformation.T)
            tree_t = KDTree(target_pair, leaf_size=10)
            dist, _ = tree_t.query(changed_pair_s, k=1)
            result_fitness = (dist <= ransac_para.max_correspondence_distance).sum() / T[3].mean()

            if max_fitness >= result_fitness:
                pass
            else:
                max_fitness = result_fitness
                best_result = T[0]
                best_pair = (np.array(T[1]), np.array(T[2])[:, :2])

            if num_validation == ransac_para.max_validation:
                break
        elif T[4] == False and (num_validation < ransac_para.max_validation):
            # update best result:
            transformation = np.delete(T[0][:2], 2, axis=-1)
            changed_pair_s = np.matmul(np.hstack((source_pair, np.ones((len(source_pair), 1)))), transformation.T)
            tree_t = KDTree(target_pair, leaf_size=10)
            dist, _ = tree_t.query(changed_pair_s, k=1)
            result_fitness = (dist <= ransac_para.max_correspondence_distance).sum() / T[3].mean()

            if max_fitness_less >= result_fitness:
                pass
            else:
                max_fitness_less = result_fitness
                best_result_less = T[0]
                best_pair_less = (np.array(T[1]), np.array(T[2])[:, :2])
    if max_fitness != -1:
        return best_result, best_pair, max_fitness
    else:
        return best_result_less, best_pair_less, max_fitness_less


def calculate_unit_normal_vector(points):
    unit_normal_vectors = []
    for i in range(len(points)):
        # 计算当前点和相邻点之间的向量
        vector1 = (points[(i+1)%len(points)][0] - points[i][0], points[(i+1)%len(points)][1] - points[i][1])
        vector2 = (points[i][0] - points[(i-1)%len(points)][0], points[i][1] - points[(i-1)%len(points)][1])
        
        # 计算法向量
        normal_vector = (vector1[1] + vector2[1], -(vector1[0] + vector2[0]))
        
        # 计算法向量的长度
        magnitude = math.sqrt(normal_vector[0]**2 + normal_vector[1]**2)
        
        # 计算单位法向量
        unit_normal_vector = (normal_vector[0]/magnitude, normal_vector[1]/magnitude) if magnitude else (0, 0)
        
        unit_normal_vectors.append(unit_normal_vector)
    return unit_normal_vectors

def calculate_vectors_sum(vectors):
    vector_sum = [0, 0]
    for vector in vectors:
        vector_sum[0] += vector[0]
        vector_sum[1] += vector[1]
    return tuple(vector_sum)

def calculate_angle(vector1, vector2):
    dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    if magnitude1 == 0 or magnitude2 == 0:
        print("Warning: One or both of the vectors are zero vectors.")
        return None
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    # 限制cos_theta在-1和1之间
    cos_theta = max(min(cos_theta, 1), -1)
    
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def iter_matchV2(
    pcd_source, pcd_target,
    proposal,
    checker_params
):
    proposal.sort()
    source_pcd, target_pcd = pcd_source[proposal], pcd_target[proposal]

    #复制用于判断向量方向的点
    S_p = source_pcd.copy()
    T_p = target_pcd.copy()

    # unit_normal_vectors1 = calculate_unit_normal_vector(S_p)
    # unit_normal_vectors2 = calculate_unit_normal_vector(T_p)
    # vector_sum1 = calculate_vectors_sum(unit_normal_vectors1[1:-1])
    # vector_sum2 = calculate_vectors_sum(unit_normal_vectors2[1:-1])

    # c. fast correspondence distance check:s
    source_pcd = np.pad(source_pcd, ((0, 0), (0, 1)), 'constant', constant_values=1)
    target_pcd = np.pad(target_pcd, ((0, 0), (0, 1)), 'constant', constant_values=1)
    T = rigid_transform_2d(source_pcd, target_pcd)  # 通过 svd 初步求解 旋转、平移矩阵
    T = np.vstack((T, np.array([0., 0., 0., 1.], dtype=np.float32)))
    R, t = T[0:3, 0:3], T[0:3, 3]

    # Source_pcd_transformed = (np.dot(source_pcd, R.T) + t)[:,:2]
    # Target_pcd = target_pcd[:,:2]
    # unit_normal_vectors1 = calculate_unit_normal_vector(Source_pcd_transformed)
    # unit_normal_vectors2 = calculate_unit_normal_vector(Target_pcd)
    # vector_sum1 = calculate_vectors_sum(unit_normal_vectors1[1:-1])
    # vector_sum2 = calculate_vectors_sum(unit_normal_vectors2[1:-1])
    # angle = calculate_angle(vector_sum1, vector_sum2)
    # # print(angle)
    # if angle is None:
    #     angle_good = False
    # elif angle < 90:
    #     angle_good = True
    # else:
    #     angle_good = False
    
    # deviation：偏差  区分 inline outline 通过 距离判断
    deviation = np.linalg.norm(
       target_pcd - np.dot(source_pcd, R.T) - t,
       axis=1
    )

    is_valid_correspondence_distance = np.all(deviation <= checker_params.max_correspondence_distance)

    return T, source_pcd, target_pcd, deviation, is_valid_correspondence_distance
    # if is_valid_correspondence_distance else None

def ransac_matchV2(source_pair, target_pair):
    distance_threshold_init = 10
    # RANSAC configuration:
    ransac_para = collections.namedtuple(
        'RANSACParams',
        [
            'max_workers',
            'num_samples',
            'max_correspondence_distance', 'max_iter', 'max_validation',
            'd'
        ]
    )
    # fast pruning algorithm configuration:
    check_para = collections.namedtuple(
        'CheckerParams',
        ['max_correspondence_distance', 'max_edge_length_ratio', 'normal_angle_threshold']
    )
    ransac_para = ransac_para(
        max_workers=8,
        num_samples=5,# 测试了4，5，6，8，还是5效果好，更深一层原因有待考证
        max_correspondence_distance=distance_threshold_init,
        max_iter=4000,
        # max_iter=3000,
        max_validation=20,
        d=15
    )


    check_para = check_para(
        max_correspondence_distance=distance_threshold_init,
        max_edge_length_ratio=0.9,
        normal_angle_threshold=None
    )

    n = len(source_pair)
    idx_matches = np.arange(n)

    validator = lambda proposal: iter_matchV2(source_pair, target_pair, proposal[0], check_para)

    max_fitness = -1
    best_result = None
    best_pair = None

    max_fitness_less = -1
    best_result_less = None
    best_pair_less = None
    # ransac:
    num_validation = 0
    for i in range(ransac_para.max_iter):
        # proposal = np.random.choice(idx_matches, ransac_para.num_samples, replace=False)
        proposal = np.arange(0, len(idx_matches))
        random.shuffle(proposal)
        proposal = proposal[:ransac_para.num_samples]
        T = validator((proposal, i))
        # if (not (T is None)) and (num_validation < ransac_para.max_validation):
        transformation = np.delete(T[0][:2], 2, axis=-1)
        changed_pair_s = np.matmul(np.hstack((source_pair, np.ones((len(source_pair), 1)))), transformation.T)
        deviation = np.linalg.norm(
            target_pair - changed_pair_s,
            axis=1
        )

        # indices = np.where(deviation < 10)
        # if len(indices[0]) > 2:
        #     Source_pcd_transformed = changed_pair_s[indices]
        #     Target_pcd = target_pair[indices]
        #     unit_normal_vectors1 = calculate_unit_normal_vector(Source_pcd_transformed)
        #     unit_normal_vectors2 = calculate_unit_normal_vector(Target_pcd)
        #     vector_sum1 = calculate_vectors_sum(unit_normal_vectors1[1:-1])
        #     vector_sum2 = calculate_vectors_sum(unit_normal_vectors2[1:-1])
        #     angle = calculate_angle(vector_sum1, vector_sum2)
        #     # print(angle)
        #     if angle is None:
        #         angle_good = False
        #     elif angle > 90:
        #         angle_good = True
        #     else:
        #         angle_good = False
        # else: angle_good = False

        # if T[4] == True and (num_validation < ransac_para.max_validation) and np.sum(deviation<10)>ransac_para.d and angle_good==True:
        if T[4] == True and (num_validation < ransac_para.max_validation) and np.sum(deviation<10)>ransac_para.d:
            num_validation += 1
            # update best result:
            transformation = np.delete(T[0][:2], 2, axis=-1)
            changed_pair_s = np.matmul(np.hstack((source_pair, np.ones((len(source_pair), 1)))), transformation.T)
            tree_t = KDTree(target_pair, leaf_size=10)
            dist, _ = tree_t.query(changed_pair_s, k=1)
            result_fitness = (dist <= ransac_para.max_correspondence_distance).sum() / T[3].mean()

            if max_fitness >= result_fitness:
                pass
            else:
                max_fitness = result_fitness
                best_result = T[0]
                best_pair = (np.array(T[1]), np.array(T[2])[:, :2])
                # best_angle = T[6]

            if num_validation == ransac_para.max_validation:
                break
        elif T[4] == False and (num_validation < ransac_para.max_validation):
            # update best result:
            transformation = np.delete(T[0][:2], 2, axis=-1)
            changed_pair_s = np.matmul(np.hstack((source_pair, np.ones((len(source_pair), 1)))), transformation.T)
            tree_t = KDTree(target_pair, leaf_size=10)
            dist, _ = tree_t.query(changed_pair_s, k=1)
            result_fitness = (dist <= ransac_para.max_correspondence_distance).sum() / T[3].mean()

            if max_fitness_less >= result_fitness:
                pass
            else:
                max_fitness_less = result_fitness
                best_result_less = T[0]
                best_pair_less = (np.array(T[1]), np.array(T[2])[:, :2])
                # best_angle_less = T[6]
    if max_fitness != -1:
        return best_result, best_pair, max_fitness
    else:
        return best_result_less, best_pair_less, max_fitness_less

