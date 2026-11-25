'''
==============================================================

    0-------------------------------0
    |       Loss Functions          |
    0-------------------------------0

==============================================================

    Compute chamfer distance loss L1/L2

==============================================================
'''

import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
MIN_NORM = 1e-15

chamfer_dist = chamfer_3DDist()

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    if pcd.shape[1] == n_points:
        return pcd
    elif pcd.shape[1] < n_points:
        raise ValueError(
            'FPS subsampling receives a larger n_points: {:d} > {:d}'.format(
                n_points, pcd.shape[1]))
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(),
        furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    # import pdb; pdb.set_trace()
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]


def get_loss1(pcds_pred, partial, gt, sqrt=True):
    # def get_loss_InfoV2(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    # cdc = CD(Pc, gt_c)
    cdc = calc_cd_like_InfoV2(Pc, gt_c)
    # cd1 = CD(P1, gt_1)
    cd1 = calc_cd_like_InfoV2(P1, gt_1)
    # cd2 = CD(P2, gt_2)
    cd2 = calc_cd_like_InfoV2(P2, gt_2)
    # cd3 = CD(P3, gt)
    cd3 = calc_cd_like_InfoV2(P3, gt)

    # partial_matching = PM(partial, P3)
    # partial_matching = calc_cd_one_side_like_hyperV2(partial, P3)
    partial_matching = calc_cd_one_side_like_InfoV2(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]


def calc_cd_like_InfoV2(p1, p2):
    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(
        torch.exp(-0.5 * d1) / (torch.sum(torch.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)
    distances2 = - torch.log(
        torch.exp(-0.5 * d2) / (torch.sum(torch.exp(-0.5 * d2) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)

    return (torch.sum(distances1) + torch.sum(distances2)) / 2


def calc_cd_one_side_like_InfoV2(p1, p2):
    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(
        torch.exp(-0.5 * d1) / (torch.sum(torch.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)

    return torch.sum(distances1)

import random
import torch.nn.functional as F
def seprate_point_cloud(xyz,
                        num_points,
                        crop,
                        fixed_points=None,
                        padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1),
                                     p=2,
                                     dim=-1)  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1,
                            descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0,
                                        idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps_subsample(input_data, 2048))
            CROP.append(fps_subsample(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()

