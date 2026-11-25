# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2
#
# import os
# script_path = os.path.dirname(os.path.abspath(__file__))
# from torch.utils.cpp_extension import load
# if torch.cuda.is_available():
#     cd = load(name="cd",
#               sources=[os.path.join(script_path, "chamfer_distance.cpp"),
#                        os.path.join(script_path, "chamfer_distance.cu")])
# class ChamferDistanceFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         batchsize, n, _ = xyz1.size()
#         _, m, _ = xyz2.size()
#         xyz1 = xyz1.contiguous()
#         xyz2 = xyz2.contiguous()
#         dist1 = torch.zeros(batchsize, n)
#         dist2 = torch.zeros(batchsize, m)
#
#         idx1 = torch.zeros(batchsize, n, dtype=torch.int)
#         idx2 = torch.zeros(batchsize, m, dtype=torch.int)
#
#         if not xyz1.is_cuda:
#             cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
#         else:
#             dist1 = dist1.cuda()
#             dist2 = dist2.cuda()
#             idx1 = idx1.cuda()
#             idx2 = idx2.cuda()
#             cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
#
#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
#
#         return dist1, dist2, idx1
#
#     @staticmethod
#     def backward(ctx, graddist1, graddist2, _):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#
#         graddist1 = graddist1.contiguous()
#         graddist2 = graddist2.contiguous()
#
#         gradxyz1 = torch.zeros(xyz1.size())
#         gradxyz2 = torch.zeros(xyz2.size())
#
#         if not graddist1.is_cuda:
#             cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
#         else:
#             gradxyz1 = gradxyz1.cuda()
#             gradxyz2 = gradxyz2.cuda()
#             cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
#
#         return gradxyz1, gradxyz2
#
# class ChamferDistance(torch.nn.Module):
#     def forward(self, xyz1, xyz2):
#         return ChamferDistanceFunction.apply(xyz1, xyz2)
#
