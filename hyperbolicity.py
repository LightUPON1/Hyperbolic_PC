# # import os
# # import pickle as pkl
# # import sys
# import time
# import networkx as nx
# import torch
# from tqdm import tqdm
# from data.data_generators import load_dataset
import numpy as np
import torch
# import torch.nn as nn
# import torchvision
from scipy.spatial import distance_matrix as scipy_Minkowski_dist_mat
from tqdm import tqdm
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#



# def hyperbolicity_sample(G, num_samples=50000):
#     curr_time = time.time()
#     hyps = []
#     for i in tqdm(range(num_samples)):
#         try:
#             curr_time = time.time()
#             node_tuple = np.random.choice(G.nodes(), 4, replace=False)
#             s = []
#             d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
#             d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
#             d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
#             d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
#             d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
#             d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
#             s.append(d01 + d23)
#             s.append(d02 + d13)
#             s.append(d03 + d12)
#             s.sort()
#             hyps.append((s[-1] - s[-2]) / 2)
#         except Exception as e:
#             continue
#     print('Time for hyp: ', time.time() - curr_time)
#     ok = True
#     if len(hyps) == 0:
#         ok = False
#         return ok,None
#     return ok,max(hyps)



# Generate a synthetic distance matrix
def generate_synthetic_points(dimensions, num_points):
    points = np.random.rand(num_points, dimensions)
    return points

def build_dist_matrix(data):
    arr_all_dist = []
    for p in data:
        arr_dist = list(
            map(lambda x: 0 if (p == x).all() else np.linalg.norm(p - x), data)
        )
        arr_all_dist.append(arr_dist)
    arr_all_dist = np.asarray(arr_all_dist)
    # arr_all_dist = torch.tensor(arr_all_dist)
    return arr_all_dist
def build_dist_matrix_torch(data):
    arr_all_dist = []
    for p in data:
        arr_dist = list(
            map(lambda x: 0 if (p == x).all() else torch.linalg.norm(p - x), data)
        )
        arr_all_dist.append(arr_dist)
    arr_all_dist = torch.tensor(arr_all_dist)
    # arr_all_dist = torch.tensor(arr_all_dist)
    return arr_all_dist

def generate_dists(dim=100, num_points=100):
    points = generate_synthetic_points(dim, num_points)
    dist_arr = build_dist_matrix(points)
    return points, dist_arr

def delta_hyp(dismat):
    # https://github.com/leymir/hyperbolic-image-embeddings/blob/6633edbbeffd6d90271f0963852a046c64f407d6/hyptorch/delta.py#L14
    """
    computes delta hyperbolicity value from distance matrix
    """
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def batched_delta_hyp(X, n_tries=10, batch_size=1500):
    # https://github.com/leymir/hyperbolic-image-embeddings/blob/6633edbbeffd6d90271f0963852a046c64f407d6/hyptorch/delta.py#L14
    """
    batched_delta_hyp: For each batch of nodes, the function computes the pairwise distances, computes the delta hyperbolicity value,
    and then aggregates the results across all batches to obtain the final delta hyperbolicity value.
    """
    vals = []
    for i in tqdm(range(n_tries)):
        idx = np.random.choice(len(X), batch_size)
        X_batch = X[idx]
        distmat = scipy_Minkowski_dist_mat(X_batch, X_batch)
        diam = np.max(distmat)
        delta_rel = 2 * delta_hyp(distmat) / diam
        vals.append(delta_rel)
    return np.mean(vals), np.std(vals)

def get_delta(all_features):
    """
    computes delta value for features;
    input -- features
    """
    all_features = all_features.reshape(all_features.shape[0], -1)
    # idx = np.random.choice(len(all_features), 50)
    # all_features_small = all_features[idx]
    dists = scipy_Minkowski_dist_mat(all_features, all_features)
    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam


# intrinsic: geometry and topology methods for DL
# https://github.com/katyafilimoshina/intrinsic/blob/6c6d818be9d1a5ab406bb060a7b5600e878d50d2/functional/delta.py#L102
def min_max_prod_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.minimum(x.unsqueeze(-1), y.unsqueeze(0)), dim=-2).values

def delta_hyperbolicity_torch(x: torch.Tensor, distances: bool = False) -> torch.Tensor:
    p = 0
    d = x if distances else torch.cdist(x, x)
    # row, col = d[:, p, :].unsqueeze(0), d[:, :, p].unsqueeze(-1)
    row, col = d[p, :].unsqueeze(0), d[:, p].unsqueeze(-1)
    A = (row + col - d) / 2
    return torch.max(torch.max(min_max_prod_torch(A, A) - A, dim=-1).values, dim=-1).values

# https://github.com/nalexai/hyperlib/blob/b7c47e121cc56b4276a387a6d8c1d05ae2445805/hyperlib/embedding/metric.py#L14
def mat_gromov_prod(dists, base):
    '''Gromov products of N-point metric space relative to base point
    Args:
        dists (ndarray): NxN matrix of pairwise distances
        base (int): index of the basepoint in 0...N-1 '''
    assert dists.shape[0] == dists.shape[1] and 0 <= base < dists.shape[0]
    row = dists[base, :][None, :]
    col = dists[:, base][:, None]
    return 0.5*(row+col-dists)

def max_min(A, B):
    '''max-min product of two square matrices
    params:
        A, B: NxN numpy arrays '''
    assert A.shape == B.shape
    return np.max(np.minimum(A[:, :, None], B[None, :, :]), axis=1)

def is_metric(X, tol=1e-8):
    return len(X.shape) == 2 and \
            np.all( np.abs(X-X.T)<tol ) and\
            np.all( np.abs(np.diag(X))<tol ) and\
            np.all(X >= 0)

def delta_rel(dists, base=None):
    ''' Measure the delta-hyperbolicity constant of data
    with respect to basepoint, normalized by the diameter (max dist).
    Args:
        dists (ndarray): NxN matrix of pairwise distances
        base (int): index of basepoint in 0...N-1 (default = random)
    '''
    from random import randint
    if base is None:
        base = randint(0,dists.shape[0]-1)
    assert is_metric(dists) and 0 <= base < dists.shape[0]
    G = mat_gromov_prod(dists, base)
    delta = np.max(max_min(G,G)-G)
    diam = np.max(dists)
    return delta/diam

# https://github.com/tnmtvv/hypdelta/blob/main/src/hypdelta/naive.py#L78
from typing import Tuple
from numba import cuda
import math
def cuda_prep_naive(
    dist_matrix: np.ndarray, threadsperblock: Tuple[int, int, int]
) -> Tuple[
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    cuda.devicearray.DeviceNDArray,
    Tuple[int, int, int],
    Tuple[int, int, int],
]:
    """
    Prepares the necessary CUDA variables for the naive delta hyperbolicity calculation.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.
    threadsperblock : Tuple[int, int, int]
        The number of threads per block.

    Returns:
    --------
    Tuple containing:
        - adj_m : cuda.devicearray.DeviceNDArray
            The distance matrix flattened and transferred to the device.
        - k : cuda.devicearray.DeviceNDArray
            An array containing parameters for the CUDA kernel.
        - diff_num_arr : cuda.devicearray.DeviceNDArray
            An array containing the difference between the nearest power of two greater than the needed number of threads
            and the nearest power of two greater than the actual number of available threads.
        - delta_res : cuda.devicearray.DeviceNDArray
            An array to store the result of the delta calculation.
        - threadsperblock : Tuple[int, int, int]
            The number of threads per block.
        - blockspergrid : Tuple[int, int, int]
            The number of blocks per grid.
    """

    # Calculate the log base 2 of the number of rows in the distance matrix, plus one
    k_1 = int(math.log2(dist_matrix.shape[0])) + 1

    # Calculate the nearest power of two greater than the number of rows in the distance matrix
    k_2 = 2**k_1 - 1

    # Get the number of rows in the distance matrix
    k_3 = dist_matrix.shape[0]

    # Transfer the flattened distance matrix to the GPU
    adj_m = cuda.to_device(dist_matrix.flatten())

    # Transfer the k parameters to the GPU
    k = cuda.to_device([k_2, k_1, k_3])

    # Allocate memory on the GPU to store the result of the delta calculation
    delta_res = cuda.to_device(np.zeros(1))

    # Calculate the difference between the nearest power of two greater than the needed number of threads
    # and the nearest power of two greater than the actual number of available threads
    diff_num = int(
        (2 * k_1)
        - math.log2(min(65536, int((2 ** (2 * k_1)) / threadsperblock[0])) * 16)
    )

    # Transfer the diff_num to the GPU
    diff_num_arr = cuda.to_device([diff_num])

    # Calculate the number of blocks per grid for each dimension, making sure it doesn't exceed 65535
    blockspergrid_x = min(65535, int((2 ** (2 * k_1)) / threadsperblock[0]) - 1)
    blockspergrid_y = min(
        65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[1])) + 1
    )
    blockspergrid_z = min(
        65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[2])) + 1
    )

    # Combine the number of blocks per grid into a tuple
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Return all prepared variables
    return adj_m, k, diff_num_arr, delta_res, threadsperblock, blockspergrid
@cuda.jit
def true_delta_gpu(
    dismat: np.ndarray, delta_res: np.ndarray, num_arr: np.ndarray, diff_num: np.ndarray
) -> None:
    """
    CUDA kernel function to compute the true delta value on the GPU.

    Parameters:
    -----------
    dismat : np.ndarray
        The distance matrix.
    delta_res : np.ndarray
        The array to store the resulting delta value.
    num_arr : np.ndarray
        An array containing configuration numbers.
    diff_num : np.ndarray
        An array containing the difference number.
    """
    num_1 = num_arr[0]
    num_2 = num_arr[1]
    d_m_len = num_arr[2]

    diff_num = diff_num[0]

    h = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) & num_1
    k = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) >> (num_2 - diff_num)
    c = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    p = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    if p < d_m_len and c < d_m_len and h < d_m_len and k < d_m_len:
        cuda.atomic.max(
            delta_res,
            (0),
            0.5
            * (
                dismat[p * d_m_len + h]
                + dismat[k * d_m_len + c]
                - max(
                    dismat[p * d_m_len + k] + dismat[h * d_m_len + c],
                    dismat[p * d_m_len + c] + dismat[k * d_m_len + h],
                )
            ),
        )

@torch.no_grad()
def delta_naive_gpu(
    dist_matrix: np.ndarray, threadsperblock: Tuple[int, int, int]=(16, 16, 4),
) -> float:
    """
    Computes the naive delta and diameter of a distance matrix using GPU.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.

    threadsperblock : Tuple[int, int, int]
    A tuple specifying the number of threads per block to use in the GPU computation.

    Returns:
    --------
    float
        Сontaining the delta.

    threadsperblock : tuple of int, optional
    The number of threads per block for GPU computation. Default is (16, 16, 4).
    """
    dist_matrix = dist_matrix.detach()
    diam = torch.max(dist_matrix)
    # adj_m, k, diff_num_arr, delta_res, threadsperblock, blockspergrid
    adj_m, k, diff_num_arr, delta_res, threadsperblock, blockspergrid = cuda_prep_naive(
        dist_matrix, threadsperblock
    )
    true_delta_gpu[blockspergrid, threadsperblock](adj_m, delta_res, k, diff_num_arr)
    delta = delta_res[0]
    return 2 * delta / diam


import util_lorentz  as L


if __name__ == '__main__':
#     # dataset_name = 'community_small'  # =1
#     # dataset_name = 'ego_small'  # (0.2525)
#     # dataset_name = 'ENZYMES'  #1.2619
#     # dataset_name = 'grid'   #9.76
#     dataset_name = 'qm9_test_nx'  #0.7091
#     # dataset_name = 'zinc250k_test_nx'  #0.9734
#     graph_list = load_dataset(file_name=dataset_name)
#     # length = len(graph_list)
#     # graph_list = graph_list[:length//10]
#
#     # graph = nx.from_scipy_sparse_matrix(data['adj_train'])
#     # print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
#     hyp_l = []
#     for i in tqdm(graph_list):
#         ok,res = hyperbolicity_sample(i,2000)
#         if not ok:
#             continue
#         hyp_l.append(res)
#         print(f'graph:{res}')
#     hyp = torch.Tensor(hyp_l).mean()
#     print('Hyp: ', hyp)
#
    all_features, distance_matrix = generate_dists(dim=10, num_points=50)
    lorentz_distance_mat = L.pairwise_dist(torch.tensor(all_features), torch.tensor(all_features))
    lorentz_distance_mat2 = L.lorentz_distance_matrix(torch.tensor(all_features))
    # delta = delta_hyp(distance_matrix)
    lorentz_distance_mat[lorentz_distance_mat< 1e-07] = 0
    # Calculate delta hyperbolicity using the naive strategy on CPU
    # delta_naive_gpu(np.array(lorentz_distance_mat))
    # delta_naive_gpu(np.array(distance_matrix))
    print(f'lorentz_distance_mat by L.pairwise_dist: {lorentz_distance_mat}')
    print(f'lorentz_distance_mat by L.lorentz_distance_matrix: {lorentz_distance_mat2}')
    print(f'difference --- lorentz_distance_mat ----: {lorentz_distance_mat - lorentz_distance_mat2}')

    distance_matrix = lorentz_distance_mat2
    print(f"delta = delta_hyp(distance_matrix): {delta_hyp(np.array(distance_matrix))}")
    print(f"delta = delta_hyp(lorentz_distance_mat): {delta_hyp(np.array(lorentz_distance_mat))}")
    print(f"delta_naive_gpu(np.array(distance_matrix)): {delta_naive_gpu(torch.tensor(distance_matrix))}")
    print(f"delta_naive_gpu(np.array(lorentz_distance_mat)): {delta_naive_gpu(torch.tensor(lorentz_distance_mat))}")
