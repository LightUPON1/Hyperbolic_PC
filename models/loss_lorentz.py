
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
import torch.nn.functional as F


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

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
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
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
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
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
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
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
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
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    if _half_aperture.isnan().any():
        print(f'NaN detected in _half_aperture')
    return _half_aperture


def exterior_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
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

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle

def lorentz_product(x, y):
    # Compute the Lorentzian inner product
    return -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)

def angle_between(x, y):
    # Compute the angle between vectors in hyperbolic space
    return torch.acos(torch.clamp(lorentz_product(x, y) / torch.sqrt(lorentz_product(x, x) * lorentz_product(y, y)), min=-1.0, max=1.0))

def hyperbolic_angle_based_loss(feature_text, feature_img, tau=0.1):
    """
    Computes the hyperbolic angle-based loss for text and image features.

    Args:
    - feature_text: Tensor of shape (B, dim), text embeddings
    - feature_img: Tensor of shape (B, dim), image embeddings
    - tau: Temperature parameter for contrastive loss

    Returns:
    - loss: The computed angle-based loss value
    """
    # Compute pairwise similarity
    similarity = pairwise_dist(feature_text, feature_img)

    # Compute the angle-based contrastive loss
    positive_pairs = torch.exp(-similarity / tau)
    negative_pairs = torch.exp(-similarity.t() / tau)

    loss_text_to_img = -torch.log(positive_pairs / torch.sum(negative_pairs, dim=1))
    loss_img_to_text = -torch.log(positive_pairs / torch.sum(negative_pairs, dim=0))

    loss = torch.mean(loss_text_to_img + loss_img_to_text)

    return loss

def hyperbolic_centroid_loss(feature_text, feature_img):
    """
    Computes the centroid loss for text and image features in hyperbolic space.

    Args:
    - feature_text: Tensor of shape (B, dim), text embeddings
    - feature_img: Tensor of shape (B, dim), image embeddings

    Returns:
    - loss: The computed centroid loss value
    """
    def to_klein_coordinates(x):
        return x[..., 1:] / x[..., 0:1]

    def to_lorentz_coordinates(k):
        norm_k = torch.norm(k, dim=-1, keepdim=True)
        return torch.cat([torch.sqrt(1 + norm_k**2), k], dim=-1)
    
    # Convert to Klein coordinates
    klein_text = to_klein_coordinates(feature_text)
    klein_img = to_klein_coordinates(feature_img)
    
    # Compute centroids in Klein coordinates
    centroid_text = torch.mean(klein_text, dim=0)
    centroid_img = torch.mean(klein_img, dim=0)
    
    # Convert back to Lorentz coordinates
    centroid_text = to_lorentz_coordinates(centroid_text)
    centroid_img = to_lorentz_coordinates(centroid_img)
    
    # Compute the Euclidean distance between centroids
    centroid_distance = torch.norm(centroid_text - centroid_img)

    Lcentroid_loss = torch.max(centroid_distance, torch.zeros_like(centroid_distance))

    # return centroid_distance
    return Lcentroid_loss

def hyperboloid_to_klein(x):
    return x[..., 1:] / x[..., 0:1]
def klein_to_hyperboloid(k, c):
    norm_k = torch.norm(k, dim=-1, keepdim=True)
    lorentz_factors = torch.sqrt(c * (1 - norm_k**2))
    x = torch.cat((torch.ones_like(k[0:1]), k), dim=-1) / lorentz_factors
    return x

def Centroid_Hyperboloid(x, c):
    k = hyperboloid_to_klein(x)
    lorentz_factors = 1 / torch.sqrt(1 - c * torch.norm(k, dim=-1) ** 2)
    weighted_projected_points = lorentz_factors[:, None] * k
    centroid_projected = torch.sum(weighted_projected_points, dim=0)
    centroid_projected /= torch.norm(lorentz_factors, dim=-1, keepdim=True)
    centroid_hyperboloid = klein_to_hyperboloid(centroid_projected, c)
    return centroid_hyperboloid

def einstein_midpoint_regularization(parent_embedding, child_embedding, curv, p=1, q=2):
    text_centroid = Centroid_Hyperboloid(parent_embedding, curv)
    image_centroid = Centroid_Hyperboloid(child_embedding, curv)
    loss_parent = torch.norm(text_centroid - torch.cosh(-curv * q)/torch.sqrt(curv))
    loss_child = torch.norm(image_centroid - torch.cosh(-curv * p)/torch.sqrt(curv))
    Lcentroid_loss = loss_parent + loss_child
    return Lcentroid_loss