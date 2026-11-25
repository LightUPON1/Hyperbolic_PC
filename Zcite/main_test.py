from hyptorch.nn import *
import torch

topoincare = ToPoincare(c=0.1, ball_dim=512, riemannian=False, clip_r=None)  # hyperbolic embedding
fusion_feat = torch.ones([3, 1, 1])
fusion_feat_poin = topoincare(fusion_feat)
print(fusion_feat_poin.shape)