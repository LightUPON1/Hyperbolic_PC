# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""

from __future__ import annotations
import math
import torch
from torch import Tensor
import warnings


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
def artanh(x):
    return Artanh.apply(x)
def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)

def _dist(x, y, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c

def dist(x, y, *, c=1.0, keepdim=False):
    # https://github.com/leymir/hyperbolic-image-embeddings/blob/6633edbbeffd6d90271f0963852a046c64f407d6/hyptorch/pmath.py
    r"""
    Distance on the Poincare ball
    .. math::
        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|(-x)\oplus_c y\|_2)
    .. plot:: plots/extended/poincare/distance.py
    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)

@torch.no_grad()
def lorentz_distance_matrix(points: torch.Tensor, curvature: float = torch.tensor(1.0)) -> torch.Tensor:
    """
    Computes the geodesic distance matrix between all points in Hyperbolic space.

    Args:
        points (torch.Tensor): A tensor of shape (N, D) where N is the number of points and D is the dimension of each point.
        curvature (float): The curvature of the hyperbolic space.

    Returns:
        torch.Tensor: The geodesic distance matrix of shape (N, N).
    """
    curvature = torch.as_tensor(curvature).type_as(points)
    # Compute pairwise distances using broadcasting
    expanded_points = points.unsqueeze(1)  # Shape: (N, 1, D)
    expanded_points_T = points.unsqueeze(0)  # Shape: (1, N, D)
    # Compute the distance matrix
    distance_matrix = dist(expanded_points, expanded_points_T, c=curvature, keepdim=True)
    # Fill the lower triangular part to ensure symmetry
    distance_matrix = torch.triu(distance_matrix) + torch.tril(distance_matrix.T, diagonal=-1)
    return distance_matrix

# https://github.com/ae-bii/neural-geometry/blob/548ef6893adf07d9a29a9e6bae01f88731844ef4/nlgm/manifolds.py#L173


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """
    x_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(y ** 2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    # xyl = np.einsum("nkd,nld->nkl", x.cpu().detach().numpy(), y.cpu().detach().numpy()) - np.einsum("nkd,nld->nkl", x_time.cpu().detach().numpy(), y_time.cpu().detach().numpy())
    return xyl

# @torch.no_grad()
def pairwise_dist(
        x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-15
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """
    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv ** 0.5

def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-15) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """
    rc_xnorm = curv ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output

# @torch.no_grad()
def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-15) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1e-15 + 1 + curv * torch.sum(x ** 2, dim=-1, keepdim=True))
    with torch.no_grad():
        _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output

# @torch.no_grad()
def half_aperture(
        x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-15
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.

    # autocast_20241008
    # with torch.autocast(x.device.type, dtype=torch.float64):
    # with torch.autocast(x.device.type, enabled=False):
    #
    # HypCLIP_20241009
    # x_space = x.narrow(-1, 1, x.shape[-1] - 1)
    x_space = x
    asin_input = 2 * min_radius / (torch.norm(x_space, dim=-1) * curv ** 0.5 + eps)
    with torch.no_grad():
        _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    # if torch.isnan(_half_aperture).any():
    #     print(f'NaN detected in _half_aperture')
    # if (_half_aperture==0).any():
    #     print(f'0 detected in _half_aperture')
    # https://github.com/facebookresearch/poincare-embeddings/blob/main/hype/manifolds/lorentz.py#L167
    return _half_aperture

# @torch.no_grad()
def exterior_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-15):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """
    # 1: Calculate time components of inputs (multiplied with `sqrt(curv)`): https://github.com/hchautran/HypCLIP/blob/7c706437a588e53b30149c01cea178077b7d26d4/hyptorch/lorentz/manifold.py#L162
    # x_space = x.narrow(-1, 1, x.shape[-1] - 1)
    # y_space = y.narrow(-1, 1, y.shape[-1] - 1)
    # x_time = x.narrow(-1, 0, 1)
    # y_time = x.narrow(-1, 0, 1)
    # 2: https://github.com/facebookresearch/meru/blob/6d4430055f58354539f68668175135256f99d47f/meru/lorentz.py#L161
    x_space = x
    y_space = y
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x_space * y_space, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x_space, dim=-1) * acos_denom + eps)
    with torch.no_grad():
        _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    '''
    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(x ** 2, dim=-1))
    y_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(y ** 2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(1e-15 + torch.clamp(c_xyl ** 2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)

    with torch.no_grad():
        _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    '''
    return _angle


def lorentz_factor_matrix(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-15):
    if x.shape[0] == 1:  # 单个样本输入
        x_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(x ** 2))
        h2k = x / x_time
        h_sqrt_part = 1 - curv * torch.norm(h2k) ** 2
        if h_sqrt_part < 0:
            print('lorentz_factor_matrix: h_sqrt_part< 0')
            warnings.warn('lorentz_factor_matrix: h_sqrt_part< 0')
        lorentz_factor = 1 / (torch.sqrt(1e-15 + h_sqrt_part) + eps)
        return lorentz_factor
    else:
        x_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(x ** 2, dim=-1)).unsqueeze(-1)
        h2k = x / x_time
        h_sqrt_part = 1 - curv * torch.norm(h2k, dim=1, keepdim=True) ** 2
        if torch.any(h_sqrt_part < 0):
            print('lorentz_factor_matrix: h_sqrt_part< 0')
            warnings.warn('lorentz_factor_matrix: h_sqrt_part< 0')
        lorentz_factor = 1 / (torch.sqrt(1e-15 + h_sqrt_part) + eps)
        return lorentz_factor


def einstein_midpoint(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-15):
    if x.shape[0] == 1:  # 单个样本输入
        return x[0]

    x_time = torch.sqrt(1e-15 + 1 / curv + torch.sum(x ** 2, dim=-1)).unsqueeze(-1)  # 3 ,1
    h2k = x / x_time
    h_sqrt_part = 1 - (-curv) * torch.norm(h2k, dim=1, keepdim=True) ** 2
    if torch.any(h_sqrt_part < 0):
        print('lorentz_factor_matrix: h_sqrt_part< 0')
        warnings.warn('lorentz_factor_matrix: h_sqrt_part< 0')
    factors = 1 / (torch.sqrt(1e-15 + h_sqrt_part) + eps)

    # factors = lorentz_factor_matrix(x, curv)
    # centroid = torch.sum(factors * x, 0) / (torch.sum(factors) + eps)
    centroid = torch.sum(factors * x / (torch.sum(factors) ** 2 + eps), 0)
    centroid_sqrt_part = (curv) * (1 - torch.norm(centroid, dim=-1) ** 2)
    if torch.any(centroid_sqrt_part < 0):
        print('einstein_midpoint: centroid_sqrt_part< 0')
        warnings.warn('einstein_midpoint: centroid_sqrt_part< 0')
    centroid = centroid / torch.sqrt(1e-15 + centroid_sqrt_part)
    return centroid


def lorentz2poincare(x, c):
    # https://github.com/snowy199/HyperbolicVAD/blob/0fb447e61608c0978f307275af165c2762bab76f/manifolds/lorentzian.py#L185
    try:
        radius = torch.sqrt(1e-15 + c)
    except:
        radius = c ** 0.5
    d = x.size(-1) - 1
    return (x.narrow(-1, 1, d) * radius) / (x.narrow(-1, 0, 1) + radius)


def poincare2lorentz(x, c):
    x_norm_square = torch.sum(x * x, dim=1, keepdim=True)
    return torch.cat((1 + x_norm_square, 2 * x), dim=1) / (1 - x_norm_square + 1e-15)


def hyperboloid2klein(x, c=1.0, dim=-1):
    r"""
    Maps points from Poincare model to Klein model
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    dim : int
        reduction dimension for operations
    Returns
    -------
    tensor
        points in Klein model, :math:`\mathbf{x}_{\mathbb{K}}=\frac{2 \mathbf{x}_{\mathbb{D}}}{1+c\left\|\mathbf{x}_{\mathbb{D}}\right\|^{2}}`
    """
    denom = 1.0 + c * x.pow(2).sum(dim, keepdim=True)
    return 2.0 * x / denom


def lorentz_factor(x, c=1.0, dim=-1, keepdim=False):
    # https://github.com/MichailChatzianastasis/WLHN/blob/1e1eb30ba3a751803eac5994bb3c1867e4c7f894/hypernn.py#L746
    # https://github.com/nlpAThits/hyfi/blob/587f1402ef941e73970ee5755377ae6a8cabbaae/hyfi/hypernn.py#L725
    """
    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorentz factor
    """
    return 1 / torch.sqrt(1e-15 + 1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def klein2hyperboloid(x, c=1.0, dim=-1):
    r"""
    Maps points from Klein model to Poincare model
    Parameters:
    ----------
    x : tensor
        point on the Klein model
    c : float|tensor
        ball negative curvature
    dim : int
        reduction dimension for operations
    Returns
    -------
    tensor
        points in Poincare: \mathbf{x}_{\mathbb{D}}=\frac{\mathbf{X}_{\mathbb{K}}}{1+\sqrt{1-c\left\|\mathbf{x}_{\mathbb{K}}\right\|^{2}}}
    """
    denom = 1.0 + torch.sqrt(1e-15 + 1.0 - c * x.pow(2).sum(dim, keepdim=True))
    return x / denom

def refined_einstein_midpoint(x, c=1.0):
    # https://github.com/MichailChatzianastasis/WLHN/blob/1e1eb30ba3a751803eac5994bb3c1867e4c7f894/hypernn.py#L746
    # https://github.com/snowy199/HyperbolicVAD/blob/0fb447e61608c0978f307275af165c2762bab76f/manifolds/lorentzian.py#L185
    # https://github.com/nlpAThits/hyfi/blob/587f1402ef941e73970ee5755377ae6a8cabbaae/hyfi/hypernn.py#L748
    # https://github.com/MySong7NLPer/HyperSiameseNet/blob/113774a865c82344f5c82ad81461323a6be78ff0/model.py#L81
    # https://github.com/huawei-noah/Pretrained-Language-Model/blob/0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9/HyperText/hyperbolic/poincare.py#L206

    r"""
    Finds the Einstein midpoint, analogue of finding average over features in Euclidean space.
    The input and output of this function are points in the Poincare ball, but internally the points are converted
    to the klein model of hyperbolic space to calculate the einstein midpoint.
    Parameters:
    ----------
    x : tensor
        point on the Poincare ball. The points are assumed to be in the last dim: B x seq_len x space_dim
    w : tensor
        weights for midpoint. B x seq_len
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        midpoint
    """
    # x_poincare = lorentz2poincare(x, c).T
    x_klein = x.T
    # x_klein = hyperboloid2klein(x_klein, c)
    factors = lorentz_factor(x_klein, c=c, keepdim=True)
    midpoint = torch.sum(factors * x_klein, dim=-1, keepdim=True) / torch.sum(factors, dim=1, keepdim=True)
    # midpoint = klein2hyperboloid(midpoint, c)
    midpoint = midpoint.T
    # midpoint = poincare2lorentz(midpoint, c)
    if x.shape[0] == 1:  # 单个样本输入
        print(f'refined_einstein_midpoint:{x - midpoint}')
    return midpoint


## https://github.com/MySong7NLPer/HyperSiameseNet/blob/113774a865c82344f5c82ad81461323a6be78ff0/model.py#L81
# eps = {torch.float32: 4e-3, torch.float64: 1e-5}
# min_norm = 1e-15
# def klein_constraint(x):
#     last_dim_val = x.size(-1)
#     norm = torch.reshape(torch.norm(x, dim=-1), [-1, 1])
#     maxnorm = (1 - eps[x.dtype])
#     cond = norm > maxnorm
#     x_reshape = torch.reshape(x, [-1, last_dim_val])
#     projected = x_reshape / (norm + min_norm) * maxnorm
#     x_reshape = torch.where(cond, projected, x_reshape)
#     x = torch.reshape(x_reshape, list(x.size()))
#     return x
#
# def klein_constraint(x):
#     last_dim_val = x.size(-1)
#     norm = torch.reshape(torch.norm(x, dim=-1), [-1, 1])
#     maxnorm = (1 - eps[x.dtype])
#     cond = norm > maxnorm
#     x_reshape = torch.reshape(x, [-1, last_dim_val])
#     projected = x_reshape / (norm + min_norm) * maxnorm
#     x_reshape = torch.where(cond, projected, x_reshape)
#     x = torch.reshape(x_reshape, list(x.size()))
#     return x
#
# def to_klein(x):
#     x_2 = torch.sum(x * x, dim=-1, keepdim=True)
#     x_klein = 2 * x / (1.0 + x_2)
#     x_klein = klein_constraint(x_klein)
#     return x_klein
#
# def proj(x, c):
#     norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), min_norm)
#     maxnorm = (1 - eps[x.dtype]) / (c ** 0.5)
#     cond = norm > maxnorm
#     projected = x / norm * maxnorm
#     return torch.where(cond, projected, x)
#
# def klein_to_poincare(x, c=1):
#     x_poincare = x / (1.0 + torch.sqrt(1.0 - torch.sum(x * x, dim=-1, keepdim=True)))
#     x_poincare = proj(x_poincare, c)
#     return x_poincare
#
# def lorentz_factors(x):
#     x_norm = torch.norm(x, dim=-1)
#     return 1.0 / (1.0 - x_norm ** 2 + min_norm)
#
# def einstein_midpoint(x, c=1):
#     x = to_klein(x)
#     x_lorentz = lorentz_factors(x)
#     x_norm = torch.norm(x, dim=-1)
#     # deal with pad value
#     x_lorentz = (1.0 - torch._cast_Float(x_norm == 0.0)) * x_lorentz
#     x_lorentz_sum = torch.sum(x_lorentz, dim=-1, keepdim=True)
#     x_lorentz_expand = torch.unsqueeze(x_lorentz, dim=-1)
#     x_midpoint = torch.sum(x_lorentz_expand * x, dim=1) / x_lorentz_sum
#     x_midpoint = klein_constraint(x_midpoint)
#     x_p = klein_to_poincare(x_midpoint, c)
#     return x_p
