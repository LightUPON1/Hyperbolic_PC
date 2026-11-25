import torch
import torch.nn as nn
import numpy as np
from .build import MODELS
from utils.logger import *
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformer import Group, Encoder, TransformerEncoder, TransformerDecoder
from models.CrossModal import TextTransformer as TextEncoder
from models.CrossModal import VisionTransformer as ImageEncoder
from timm.models.layers import trunc_normal_
import math
import models.lorentz as L
# from models.loss_lorentz import hyperbolic_centroid_loss, einstein_midpoint_regularization
# from extensions.infoCD import seprate_point_cloud, get_loss
# from hyptorch.nn import ToPoincare, HyperbolicDistanceLayer
from hutil import hype_triplet_losses, get_children_np
# from manifolds import PoincareBall
# from manifolds.ops.manifold_layers import MobiusLayer
# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.img_token, std=.02)
        trunc_normal_(self.text_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.img_pos, std=.02)
        trunc_normal_(self.text_pos, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, points, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        x = torch.cat((cls_token, img_token, text_token, x_vis), dim=1)
        pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)

        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        return x[:, 0], x[:, 1], x[:, 2], x[:, 3:], bool_masked_pos


class ContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, similarity, select):
        B = similarity.size(0)
        losses = torch.zeros(B).to(similarity.device)
        for i in range(B):
            pos = torch.masked_select(similarity[i], select[i] == 1)
            neg = torch.masked_select(similarity[i], select[i] == 0)
            pos = torch.mean(pos, dim=0, keepdim=True)
            logits = torch.cat((pos, neg)).reshape(1, -1)
            label = torch.zeros(1, dtype=torch.long).to(similarity.device)
            losses[i] = self.criterion(logits/self.temperature, label)
        loss = losses.mean()
        return loss


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


@MODELS.register_module()
class ReCon_hyper(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[ReCon] ', logger='ReCon')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[ReCon] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='ReCon')

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        # cross model contrastive
        self.csc_loss = torch.nn.SmoothL1Loss()
        self.csc_img = True if config.img_encoder else False
        self.csc_text = True if config.text_encoder else False

        self.embed_dim = 768
        if self.csc_img:
            self.img_encoder = ImageEncoder(config)
            for p in self.img_encoder.parameters():
                p.requires_grad = False
            self.img_proj = nn.Linear(self.trans_dim, self.img_encoder.output_dim)
            self.img_proj.apply(self._init_weights)
            # self.img_proj_unlock = nn.Linear(self.img_encoder.embed_dim, self.img_encoder.embed_dim)
            self.visual_proj = nn.Linear(self.img_encoder.embed_dim,  self.embed_dim, bias=False)
            nn.init.normal_(self.visual_proj.weight, std=self.img_encoder.embed_dim**-0.5)
            # if self.hyper_flag:
            #     visual_curv_init = 1.0
            #     self.visual_curv = nn.Parameter(
            #         torch.tensor(visual_curv_init).log(), requires_grad=True
            #     )
            #     self.visual_manifold = PoincareBall(c=self.visual_curv, dim=self.img_encoder.embed_dim)
            #     self.visual_proj = MobiusLayer(self.img_encoder.embed_dim, self.embed_dim, self.visual_manifold)
            ## self.visual_proj.apply(self._init_weights)

        if self.csc_text:
            self.text_encoder = TextEncoder(config)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_proj = nn.Linear(self.trans_dim, self.text_encoder.embed_dim)
            self.text_proj.apply(self._init_weights)
            self.textual_proj = nn.Linear(self.text_encoder.embed_dim,  self.embed_dim, bias=False)
            nn.init.normal_(self.textual_proj.weight, std=self.text_encoder.embed_dim**-0.5)
            # if self.hyper_flag:
            #     text_curv_init = 1.0
            #     self.text_curv = nn.Parameter(
            #         torch.tensor(text_curv_init).log(), requires_grad=True
            #     )
            #     self.text_manifold = PoincareBall(c=self.text_curv_init, dim=self.text_encoder.embed_dim)
            #     self.text_proj = MobiusLayer(self.text_encoder.embed_dim, self.embed_dim, self.text_manifold)
            ## self.text_proj_unlock.apply(self._init_weights)

        # single modal contrastive
        self.smc = config.self_contrastive
        if self.smc:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.trans_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128)
            )
            self.cls_proj.apply(self._init_weights)
            self.contrastive_head = ContrastiveHead(temperature=0.1)
        self.hyper_flag = True
        if self.hyper_flag:
            self.pts_proj = nn.Linear(self.trans_dim,  self.embed_dim, bias=False)
            nn.init.normal_(self.pts_proj.weight, std=self.trans_dim**-0.5)
            self.local_grouper_list = nn.ModuleList()
            self.local_grouper_list.append(self.group_divider)
            self.local_grouper_list.append(Group(num_group=self.num_group//2, group_size=self.group_size)) # self.group_size=32;  num_group == 64
            self.local_grouper_list.append(Group(num_group=self.num_group//2, group_size=self.group_size))
            curv_init = 1.0
            self.curv = nn.Parameter(
                torch.tensor(curv_init).log(), requires_grad=True
            )
            self._curv_minmax = {
                "max": math.log(curv_init * 10),
                "min": math.log(curv_init / 10),
            }
            # self.topoincare = ToPoincare(c=curv_init, train_c = True, ball_dim=self.trans_dim, riemannian=True, clip_r=None) # hyperbolic embedding
            # from manifolds import PoincareBall
            # from ops.manifold_layers import MobiusLayer
            # self.manifold = PoincareBall(c=curv_init, dim=self.trans_dim)
            # self.mobiusemb = MobiusLayer(self.trans_dim, self.trans_dim, self.manifold)

            # learn_curv = True
            # # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
            # self.curv = nn.Parameter(
            #     torch.tensor(curv_init).log(), requires_grad=learn_curv
            # )
            # # When learning the curvature parameter, restrict it in this interval to
            # # prevent training instability.
            # self._curv_minmax = {
            #     "max": math.log(curv_init * 10),
            #     "min": math.log(curv_init / 10),
            # }
            # Learnable scalars to ensure that image/text features have an expected
            # unit norm before exponential map (at initialization).
            self.visual_alpha = nn.Parameter(torch.tensor(self.embed_dim ** -0.5).log())
            self.textual_alpha = nn.Parameter(torch.tensor(self.embed_dim ** -0.5).log())
            self.pts_alpha = nn.Parameter(torch.tensor(self.embed_dim ** -0.5).log())
            self.parent_alpha = nn.Parameter(torch.tensor(self.embed_dim ** -0.5).log())
            self.child_alpha = nn.Parameter(torch.tensor(self.embed_dim ** -0.5).log())
            # self.poincar_distance = HyperbolicDistanceLayer(c=torch.tensor(curv_init))
            # self.topoincare = ToPoincare(c=torch.tensor(curv_init),
            #                              train_c=True,
            #                              ball_dim=self.embed_dim,
            #                              riemannian=True,
            #                              clip_r=None)
            self.entail_weight = 0.03
            self.centroid_weight = 0.0001
            self.t_loss_weight = 0.01
            self.h_loss_weight = 0.001
            # Initialize a learnable logit scale parameter.
            self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts, img, text, **kwargs):
        losses = {}
        loss = None
        ptss = []
        ptss.append(pts)
        if self.hyper_flag:
            innerloops = 1
            if innerloops == 3:
                _, parent_data, n_parent_points = get_children_np(pts.permute(0, 2, 1), kmin=800, kmax=1024) #, k = 768)
                # pts [btach, points, xyz] -- data.shape-torch.Size([128, 3, 815]); n_points-814 ---- pos_data.shape-torch.Size([128, 3, 215])
                # data, n_points = pts.permute(0, 2, 1), pts.shape[1] - 1
                mar, child_data, _ = get_children_np(parent_data, starting=n_parent_points, kmin=200, kmax=n_parent_points // 2)#, k=256)
                parent_data = parent_data.permute(0, 2, 1).contiguous()
                child_data = child_data.permute(0, 2, 1).contiguous()
                ptss.append(parent_data)
                ptss.append(child_data)
                ## visualization
                # import open3d as o3d
                # vis = o3d.visualization.Visualizer()
                # vis.create_window(width=1200, height=600)
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(data.permute(0, 2, 1)[0])
                # vis.add_geometry(pcd)
                # vis.run()

        for iter_innerloop in range(innerloops):
            pts = ptss[iter_innerloop]
            # neighborhood, center = self.group_divider(pts)
            neighborhood, center = self.local_grouper_list[iter_innerloop](pts)
            cls_token, img_token, text_token, x_vis, mask = self.MAE_encoder(pts, neighborhood, center)
            # torch.Size([128, 384])
            # x_vis/mask - torch.Size([128, 26, 384])/torch.Size([128, 64])
            # if self.hyper_flag:
            #     # x_vis = self.topoincare(x_vis)
            #     with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32):
            #         x_vis = L.exp_map0(x_vis * self.visual_alpha.exp(), self.curv.exp())

            B, _, C = x_vis.shape  # B VIS C
            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            _, N, _ = pos_emd_mask.shape # torch.Size([128, 38, 384])
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            ## TODO: regulization
            if self.hyper_flag:
                ## TODO: parent_mu
                if iter_innerloop == 1:
                    parent_mu = x_full
                    # parent_mu = self.topoincare(parent_mu)
                    parent_mu = self.pts_proj(torch.mean(parent_mu, dim=1))
                    parent_mu = L.exp_map0(parent_mu * self.parent_alpha.exp(), self.curv.exp())
                ## TODO: child_mu
                if iter_innerloop == 2:
                    child_mu = x_full
                    # child_mu = self.topoincare(child_mu)
                    child_mu = self.pts_proj(torch.mean(child_mu, dim=1))
                    child_mu = L.exp_map0(child_mu * self.child_alpha.exp(), self.curv.exp())

            x_rec = self.MAE_decoder(x_full, pos_full, N)
            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            losses['mdm'] = self.loss_func(rebuild_points, gt_points)
            ## TODO: infoCD-HyperbolicChamfer
            # partial, _ = seprate_point_cloud(gt_points, gt_points.shape[1], [int(gt_points.shape[1] * 1 / 4), int(gt_points.shape[1] * 3 / 4)],
            #                                                fixed_points=None)
            # losses['mdm'] = self.loss_func(rebuild_points, partial, gt_points, sqrt=True)
            if self.csc_img and iter_innerloop == 0:
                img_feature = self.img_encoder(img)
                img_token = self.img_proj(img_token)
                if self.hyper_flag:
                    img_feature = self.visual_proj(img_feature)
                    img_token = self.visual_proj(img_token)
                    with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32, enabled = True):
                        # img_feature = self.topoincare(img_feature)
                        # img_token = self.topoincare(img_token)
                        img_feature = L.exp_map0(img_feature * self.visual_alpha.exp(), self.curv.exp())
                        img_token = L.exp_map0(img_token * self.visual_alpha.exp(), self.curv.exp())
                losses['csc_img'] = self.csc_loss(img_feature, img_token).mean()
            if self.csc_text:
                if iter_innerloop == 0:
                    # text = tuple('Point cloud of ' + text_i for text_i in text)
                    text_feature = self.text_encoder(text)
                else:
                    textC = tuple('part of ' + text_i for text_i in text)
                    text_feature = self.text_encoder(textC)

                text_token = self.text_proj(text_token)

                if self.hyper_flag:
                    text_feature = self.textual_proj(text_feature)
                    text_token = self.textual_proj(text_token)
                    with torch.autocast(self.textual_alpha.device.type, dtype=torch.float32, enabled = True):
                        # text_feature = self.topoincare(text_feature)
                        # text_token = self.topoincare(text_token)
                        text_feature = L.exp_map0(text_feature * self.textual_alpha.exp(), self.curv.exp())
                        text_token = L.exp_map0(text_token * self.textual_alpha.exp(), self.curv.exp())
                losses['csc_text'] = self.csc_loss(text_feature, text_token).mean()

            if self.smc:
                # TODO:hyperbolichead
                cls_proj = self.cls_proj(cls_token)
                # if self.hyper_flag:
                #     cls_proj = self.topoincare(cls_proj)
                cls_proj = nn.functional.normalize(cls_proj, dim=1)
                similarity = torch.matmul(cls_proj, cls_proj.permute(1, 0))

                select = torch.zeros([B, B], dtype=torch.uint8).to(similarity.device)
                for i in range(B):
                    for j in range(B):
                        if text[i] == text[j]:
                            select[i, j] = 1
                losses['smc'] = self.contrastive_head(similarity, select)

            if self.hyper_flag:
                # _curv = self.topoincare.c
                self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
                _curv = self.curv.exp()

                # x_full = self.pts_proj(torch.mean(x_full, dim=1))
                x_full = self.pts_proj(torch.nn.functional.adaptive_max_pool1d(x_full.transpose(-1,-2), 1).squeeze(dim=-1))
                x_full = L.exp_map0(x_full * self.pts_alpha.exp(), self.curv.exp())

                i2t_logits = -L.pairwise_dist(img_feature, text_feature, _curv)
                t2i_logits = -L.pairwise_dist(text_feature, img_feature, _curv)
                pc2t_logits = -L.pairwise_dist(text_feature, img_feature, _curv)

                batch_size = img_feature.shape[0]
                targets = torch.arange(batch_size, device=i2t_logits.device)
                self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
                _scale = self.logit_scale.exp()

                losses['cms'] = 0.5 * (
                        nn.functional.cross_entropy(_scale * i2t_logits, targets)
                        + nn.functional.cross_entropy(_scale * t2i_logits, targets)
                        + nn.functional.cross_entropy(_scale * pc2t_logits, targets)
                )

                # Hyperbolic entailment loss: text should entail matching image.
                _angle_text_pts = L.exterior_angle(text_feature, x_full, _curv)
                _aperture_text_pts = L.half_aperture(text_feature, _curv)

                _angle_pts_img = L.exterior_angle(x_full, img_feature, _curv)
                _aperture_pts_img = L.half_aperture(x_full, _curv)

                _angle_text_img = L.exterior_angle(text_feature, img_feature, _curv)
                _aperture_text_img = L.half_aperture(text_feature, _curv)

                losses['entailment_loss'] = (self.entail_weight *
                                             (
                                                torch.clamp(_angle_text_pts - _aperture_text_pts, min=0).mean()
                                              + torch.clamp(_angle_pts_img - _aperture_pts_img, min=0).mean()
                                              + torch.clamp(_angle_text_img - _aperture_text_img, min=0).mean()
                                              )
                                             )
                # losses['cetroid_loss_1'] = (self.centroid_weight *
                #                             (einstein_midpoint_regularization(text_feature, img_feature, _curv)
                #                              +einstein_midpoint_regularization(text_feature, x_full, _curv)
                #                              )
                #                             )
            if loss == None:
                loss = sum(losses.values())
            else:
                loss += sum(losses.values())
                if iter_innerloop == 2:
                    _angle_pts = L.exterior_angle(parent_mu, child_mu, _curv)
                    _aperture_pts = L.half_aperture(parent_mu, _curv)
                    losses['h_t_loss'] = self.entail_weight * torch.clamp(_angle_pts - _aperture_pts, min=0).mean()
                    # losses['cetroid_loss_2'] = self.centroid_weight * hyperbolic_centroid_loss(parent_mu, child_mu)
                    # pn, posn, pd, nd, t_loss, h_loss = hype_triplet_losses(parent_mu, child_mu, hier_margin=mar * 0.5, contr_margin=4, ball_dim=self.trans_dim)
                    # losses['h_t_loss'] += self.t_loss_weight * t_loss + self.h_loss_weight * h_loss
                    loss = loss/3 + losses['h_t_loss'] + losses['cetroid_loss_2']
                    # loss = loss/3 + 0
                    # tensor(0.1148)
                    # tensor(3.9609, device='cuda:0', grad_fn=<MeanBackward0>)
                    # tensor(1.5480, device='cuda:0', grad_fn=<MeanBackward0>)
                    # print(f'iter_innerloop-{iter_innerloop}-loss:{loss}@h_loss：{h_loss}')
        # {'mdm': tensor(0.9354, device='cuda:0', grad_fn=<AddBackward0>),
        #  'csc_img': tensor(0.0002, device='cuda:0', grad_fn=<MeanBackward0>),
        #  'csc_text': tensor(0.0002, device='cuda:0', grad_fn=<MeanBackward0>),
        #  'entailment_loss': tensor(0.5841, device='cuda:0', grad_fn=<MulBackward0>),
        #  'h_t_loss': tensor(0.4006, device='cuda:0', grad_fn=<AddBackward0>)}

        #  single-modal contrastive models (SMC)
        #  cross-modal contrastive models (CMC)
        #  masked poinossest modeling reconstruction (MPM)
        # print(losses)
        # {'mdm': tensor(0.1130, device='cuda:0', grad_fn=<AddBackward0>),
        # 'csc_img': tensor(0.0887, device='cuda:0', grad_fn=<MeanBackward0>),
        # 'csc_text': tensor(2.1069e-05, device='cuda:0', grad_fn=<MeanBackward0>),
        # 'cms': tensor(0.4160, device='cuda:0', grad_fn=<MulBackward0>),
        # 'entailment_loss': tensor(0.0209, device='cuda:0', grad_fn=<MulBackward0>)}
        return loss