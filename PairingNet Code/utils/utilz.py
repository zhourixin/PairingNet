import numpy as np
from scipy.spatial.distance import cdist


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

    t = cdist(source_pcd_trans, target_pcd)
    source_ind = np.arange(0, len(source_pcd))
    target_ind = np.argmin(t, axis=-1)
    d = np.min(t, axis=-1)
    mask = (d <= 3.5)
    source_ind = source_ind[mask]
    target_ind = target_ind[mask]

    return source_ind, target_ind


def rigid_transform_2d(Aa, Bb):

    A = np.tile(Aa, 1)
    B = np.tile(Bb, 1)
    max = 0
    for i in range(1):
        # if len(A)<=2:
        #     break
        assert len(A) == len(B)
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # centre the points
        AA = A - np.tile(centroid_A, 1)
        BB = B - np.tile(centroid_B, 1)

        H = np.matmul(np.transpose(AA),BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.matmul(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            # print("Reflection detected")
            Vt[1, :] *= -1
            R = np.matmul(Vt.T,U.T)

        t = -np.matmul(R, centroid_A) + centroid_B

    return np.hstack((R, t.reshape(-1, 1)))


def affine_transform(pt, t):
    """pt: [n, 2]"""
    pt = np.matmul(np.hstack((pt, np.ones((len(pt), 1)))), t.T)
    return pt
