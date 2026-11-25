import torch
import torch.nn as nn
import numpy as np
from .build import MODELS
from utils.logger import *
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformer import Group, Encoder, TransformerEncoder, TransformerDecoder
# from models.meru_CrossModal import TextTransformer as TextEncoder
# from models.meru_CrossModal import VisionTransformer as ImageEncoder
from models.meru_CrossModal import MERU_pretrained_model
from models.meru.tokenizer import Tokenizer
from timm.models.layers import trunc_normal_
# import math
import models.lorentz as L
from hyperbolicity import delta_naive_gpu
# from models.loss_lorentz import hyperbolic_centroid_loss, einstein_midpoint_regularization
# from extensions.infoCD import seprate_point_cloud, get_loss
# from hyptorch.nn import ToPoincare, HyperbolicDistanceLayer
# from hutil import hype_triplet_losses, get_children_np
# from torch.nn import functional as F
# from manifolds import PoincareBall
from nltk.corpus import wordnet as wn
from models.AutomaticWeightedLoss import AutomaticWeightedLoss
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
# from timm.models.layers import DropPath

# from manifolds.ops.manifold_layers import MobiusLayer
# Pretrain model
from utils import misc

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

    def forward(self, points, neighborhood, center, noaug=False, mask_config_block=False):
        # neighborhood.shape--torch.Size([128, 12, 32, 3])
        # center.shape--torch.Size([128, 12, 3])
        # generate mask
        group_input_tokens = self.encoder(neighborhood)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()

        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        if mask_config_block == False:
            if self.mask_type == 'rand':
                bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
            else:
                bool_masked_pos = self._mask_center_block(center, noaug=noaug)
    
            x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
            # add pos embedding
            # mask pos center
            masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
            # pos = self.pos_embed(masked_center)
            pos = self.pos_embed(masked_center-masked_center[:, 0:1, :])

            x = torch.cat((cls_token, img_token, text_token, x_vis), dim=1)
            pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)

        else:
            x_vis = group_input_tokens.reshape(batch_size, -1, C)
            center = center.reshape(batch_size, -1, 3)

            # pos = self.pos_embed(center)
            pos = self.pos_embed(center-center[:, 0:1, :])

            x = torch.cat((cls_token, img_token, text_token, x_vis), dim=1)
            pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)
            bool_masked_pos = None
        # transformer
        # TransformerEncoder--blocks--Block--attn_contrast(detach)
        x = self.blocks(x, pos)
        x = self.norm(x)
        '''
        # --x--
        # x[:, 0], x[:, 1], x[:, 2], x[:, 3:], bool_masked_pos
        # cls_token, img_token, text_token, x_vis, mask
        '''
        return x, bool_masked_pos



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

class Angle_ContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1):
        super(Angle_ContrastiveHead, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.SmoothL1Loss()
        self.temperature = temperature

    def forward(self, text_feats, image_feats, _curv):
        '''
        # For MERU, exclude text embeddings that do not entail the given image.
        _aper = L.half_aperture(text_feats, model.curv.exp())
        _exterior_angle = L.exterior_angle(
            text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        )
        entailment_energy = _exterior_angle - _aper[..., None]
        '''
        # from code, L.oxy is the exterior angle not oxy angle
        similarities_exterior_alpha = - L.exterior_angle(text_feats, image_feats, _curv)
        similarities_beta = torch.tensor(torch.pi).to(text_feats.device) + similarities_exterior_alpha
        losses = []
        for simi_matrix in [similarities_exterior_alpha, similarities_beta]:
            simi_matrix = torch.exp(simi_matrix / self.temperature)
            loss = torch.log(simi_matrix/simi_matrix.sum() + 1e-15)
            loss = - loss.sum().mean()
            # label = torch.zeros(1, dtype=torch.long).to(simi_matrix.device)
            # loss = self.criterion(simi_matrix, label)
            losses.append(loss)
        return sum(losses)/len(losses)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

# from ReCon++ self.distillation_loss(token, feature)
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
def contrast_loss(token, feature):
    l1_loss = torch.nn.SmoothL1Loss()
    # return self.info_nce_loss(token, feature, logit_scale=self.logit_scale, mask=self.mask)
    B = token.shape[0]
    loss = 0.0
    for i in range(B):
        pred = token[i]
        feat = feature[i][torch.any(feature[i] != 0, dim=-1)]
        feat = F.normalize(feat, dim=-1)
        similarity_matrix = torch.mm(pred.unsqueeze(0), feat.T).cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        loss = loss + l1_loss(pred[row_ind], feat[col_ind])
    loss = loss * 5
    return loss

@MODELS.register_module()
class ReCon_meru(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[ReCon] ', logger='ReCon')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        # self.NMAE_encoder = PointTransformer(config)
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
        self.pos_group_divider = Group(num_group=12, group_size=self.group_size)

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
        # self.csc_loss = contrast_loss
        self.csc_img = True if config.img_encoder else False
        self.csc_text = True if config.text_encoder else False

        self.curv = nn.Parameter(
            MERU_pretrained_model.curv, requires_grad=False
        )
        # requires_grad = True 92.0989 epoch-101
        # requires_grad = False 92.7877 epoch-203
        # requires_grad = False 93.1118 epoch-241 92.7877 epoch-205
        alpha_requires_grad = False
        self.visual_alpha = nn.Parameter(MERU_pretrained_model.visual_alpha, requires_grad=alpha_requires_grad)
        self.textual_alpha = nn.Parameter(MERU_pretrained_model.textual_alpha, requires_grad=alpha_requires_grad)
        self.pts_alpha = nn.Parameter(torch.tensor(MERU_pretrained_model.embed_dim ** -0.5).log(), requires_grad=alpha_requires_grad)
        self._curv_minmax = MERU_pretrained_model._curv_minmax
        # meru_parametersList = [self.curv, self.visual_alpha, self.textual_alpha]
        # for meru_parameters in meru_parametersList:
        #     meru_parameters.requires_grad = False
        self.pts_proj = nn.Linear(self.trans_dim, MERU_pretrained_model.embed_dim)
        self.pts_proj.apply(self._init_weights)
        # self.img_encoder = MERU_pretrained_model.encode_image
        # self.text_encoder = MERU_pretrained_model.encode_text
        self.img_encoder = MERU_pretrained_model.visual
        self.text_encoder = MERU_pretrained_model.textual
        self.tokenizer = Tokenizer()

        if self.csc_img:
            self.visual_proj = MERU_pretrained_model.visual_proj
            meru_parametersList = [self.img_encoder, self.visual_proj]
            for meru_parameters in meru_parametersList:
                for p in meru_parameters.parameters():
                    p.requires_grad = False
            self.img_proj = nn.Linear(self.trans_dim, MERU_pretrained_model.embed_dim)
            self.img_proj.apply(self._init_weights)

        if self.csc_text:
            self.textual_proj = MERU_pretrained_model.textual_proj
            meru_parametersList = [self.text_encoder, self.textual_proj]
            for meru_parameters in meru_parametersList:
                for p in meru_parameters.parameters():
                    p.requires_grad = False
            self.text_proj = nn.Linear(self.trans_dim, MERU_pretrained_model.embed_dim)
            self.text_proj.apply(self._init_weights)

        # single modal contrastive
        # self.smc = config.self_contrastive
        # if self.smc:
        #     self.cls_proj = nn.Sequential(
        #         nn.Linear(self.trans_dim, 128),
        #         nn.BatchNorm1d(128),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(128, 128),
        #         nn.BatchNorm1d(128)
        #     )
        #     self.cls_proj.apply(self._init_weights)
        #     self.contrastive_head = ContrastiveHead(temperature=0.1)

        self.exterior_angle_contra_head = Angle_ContrastiveHead(temperature=0.1)
        self.template_part = 'Part of '
        self.template = ['',
                         'A ',
                         'A model of ',
                         'A model of a ',
                         'A image of ',
                         'A image of a ',
                         'A 3D model of ',
                         'A 3D model of a ',
                         'A rendering model of ',
                         'A rendering model of a ',
                         'A point cloud of ',
                         'A point cloud of a ',
                         'A point cloud model of ',
                         'A point cloud model of a ',
                         'A 3D rendering model of ',
                         'A 3D rendering model of a ',
                         'A rendering image of ',
                         'A rendering image of a ',
                         'A 3D rendering image of ',
                         'A 3D rendering image of a '
                         ]
        self.template_last = ['.', ' with white background.', ' with black context.']
        self.ntlk_dict = dict()
        # self.awl = AutomaticWeightedLoss(5)  # we have 2 losses
        self.uncertainty = nn.Parameter(
            torch.ones(7), requires_grad=True
        )
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def encode_image(self, images: torch.Tensor):
        self.register_buffer("pixel_mean", torch.tensor(MERU_pretrained_model.pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(MERU_pretrained_model.pixel_std).view(-1, 1, 1))
        # self.pixel_mean = (0.485, 0.456, 0.406)
        # self.pixel_std = (0.229, 0.224, 0.225)
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.img_encoder(images)
        image_feats = self.visual_proj(image_feats)
        image_feats = image_feats * self.visual_alpha.exp()
        with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32, enabled = True):
            image_feats = L.exp_map0(image_feats, self.curv.exp())
        return image_feats

    def encode_text(self, text):
        if isinstance(text, str):
            index = 0
            prompt_text = self.template[index % 20] + text + self.template_last[index // 20]
        else:
            prompt_text = [random.choice(self.template) + t + random.choice(self.template_last) for t in text]
        caption_tokens = self.tokenizer(prompt_text)
        tokens = caption_tokens
        # : list[torch.Tensor]
        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.text_encoder.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.text_encoder.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens
        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.textual_alpha.device)
        # shape: (batch_size, context_length, textual.width)
        text_feats = self.text_encoder(tokens)
        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        text_feats = text_feats * self.textual_alpha.exp()
        with torch.autocast(self.textual_alpha.device.type, dtype=torch.float32, enabled = True):
            text_feats = L.exp_map0(text_feats, self.curv.exp())
        return text_feats

    def encode_text_part(self, text):
        # if isinstance(text, str):
        #     index = 0
        #     prompt_text = self.template[index % 20] + text + self.template_last[index // 20]
        # else:
        #     prompt_text = [self.template_part + random.choice(self.template) + t + random.choice(self.template_last) for t in text]
        prompt_text = [self.template_part + random.choice(self.template) + t + random.choice(self.template_last) for t in text]
        caption_tokens = self.tokenizer(prompt_text)
        tokens = caption_tokens
        # : list[torch.Tensor]
        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.text_encoder.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.text_encoder.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens
        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.textual_alpha.device)
        # shape: (batch_size, context_length, textual.width)
        text_feats = self.text_encoder(tokens)
        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        text_feats = text_feats * self.textual_alpha.exp()
        with torch.autocast(self.textual_alpha.device.type, dtype=torch.float32, enabled = True):
            text_feats = L.exp_map0(text_feats, self.curv.exp())
        return text_feats


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


    def forward_features(self, pts, neighborhood, center, mask_config_block):
        global_token, mask = self.MAE_encoder(pts, neighborhood, center, mask_config_block)
        cls_token, img_token, text_token, x_vis = global_token[:, 0], global_token[:, 1], global_token[:, 2], global_token[:, 3:]
        # generated_points, gt_points, global_token = self.model(pts)
        if mask_config_block == False:
            B, _, C = x_vis.shape  # B VIS C
            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            _, N, _ = pos_emd_mask.shape
            if torch.isnan(self.mask_token).any():
                print('NaN detected in mask_token')

            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
            x_rec = self.MAE_decoder(x_full, pos_full, N)
            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        else:
            # mask_config_block == True
            # img_token, text_token, x_full, _, _ = ...mask_config_block=True
            #
            x_full = x_vis
            return img_token, text_token, x_full, None, None
        return img_token, text_token, x_full, gt_points, rebuild_points

    def forward(self, pts, img, text, **kwargs):
        B = pts.shape[0]
        losses = {}
        neighborhood, center = self.group_divider(pts)
        # cls_token, img_token, text_token, x_vis, mask = self.MAE_encoder(pts, neighborhood, center)
        # B, _, C = x_vis.shape  # B VIS C
        # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        # _, N, _ = pos_emd_mask.shape
        # if torch.isnan(self.mask_token).any():
        #     print('NaN detected in mask_token')
        #
        # mask_token = self.mask_token.expand(B, N, -1)
        # x_full = torch.cat([x_vis, mask_token], dim=1)
        # pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        #
        # x_rec = self.MAE_decoder(x_full, pos_full, N)
        #
        # B, M, C = x_rec.shape
        # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
        #
        # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        ### mask_config_block=True--the mask config wont work, hence no mask
        # img_token, text_token, x_full, _, _ = self.forward_features(pts, neighborhood, center, mask_config_block=True)
        img_token, text_token, x_full,gt_points, rebuild_points = self.forward_features(pts, neighborhood, center, mask_config_block=False)
        # img_token_part, text_token_part, x_part, gt_points, rebuild_points = self.forward_features(pts, neighborhood, center, mask_config_block=False)
        for token_part in [img_token, text_token, x_full, gt_points, rebuild_points]:
        # for token_part in [img_token, text_token, x_full, img_token_part, text_token_pa1rt, x_part, gt_points, rebuild_points]:
            if torch.isnan(token_part).any():
                print(f'NaN detected in token')

        losses['mdm'] = self.loss_func(rebuild_points, gt_points)

        if self.csc_img:
            img_feature = self.encode_image(img)
            #######################################################################
            # image_feats = self.visual_proj(image_feats)
            # image_feats = image_feats * self.visual_alpha.exp()
            # with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32):
            #     image_feats = L.exp_map0(image_feats, self.curv.exp())
            #######################################################################
            img_token = self.img_proj(img_token)
            # 20241025 ref ReConV2 F.normalize(img_token, dim=-1)
            # img_token = F.normalize(img_token, dim=-1)
            img_token = img_token * self.visual_alpha.exp()
            with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32, enabled = True):
                img_token = L.exp_map0(img_token, self.curv.exp())
            losses['csc_img'] = self.csc_loss(img_feature, img_token).mean()

        if self.csc_text:
            text_feature = self.encode_text(text)
            # text_feature = self.text_encoder(caption_tokens, True)
            text_token = self.text_proj(text_token)
            # text_token = F.normalize(text_token, dim=-1)
            text_token = text_token * self.textual_alpha.exp()
            with torch.autocast(self.textual_alpha.device.type, dtype=torch.float32, enabled = True):
                text_token = L.exp_map0(text_token, self.curv.exp())
            losses['csc_text'] = self.csc_loss(text_feature, text_token).mean()

        # if self.smc:
        #     cls_proj = self.cls_proj(cls_token)
        #     cls_proj = nn.functional.normalize(cls_proj, dim=1)
        #     similarity = torch.matmul(cls_proj, cls_proj.permute(1, 0))
        #
        #     select = torch.zeros([B, B], dtype=torch.uint8).to(similarity.device)
        #     for i in range(B):
        #         for j in range(B):
        #             if text[i] == text[j]:
        #                 select[i, j] = 1
        #     losses['smc'] = self.contrastive_head(similarity, select)

        # Hyperbolic entailment loss: point_clouds should entail matching image.
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        # x_full = misc.fps(x_full.contiguous(), 1)
        x_full = x_full.mean(1)
        x_full_hyperbolic = self.pts_proj(x_full)
        x_full_hyperbolic = x_full_hyperbolic * self.pts_alpha.exp()
        with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32, enabled = True):
            x_full_hyperbolic = L.exp_map0(x_full_hyperbolic, self.curv.exp())

        # full_size = neighborhood.shape[1]
        # pos_neighborhood, pos_center = neighborhood[:, 0:full_size//2, :], center[:, 0:full_size//2, :]
        # _, _, text_token_part, pos_vis, _ = self.MAE_encoder(pts, pos_neighborhood, pos_center)
        # pos_mu = torch.cat([pos_vis,  self.mask_token.clone().expand(B, pos_vis.shape[1], -1)], dim=1)
        # pos_mu_hyperbolic = self.pts_proj(pos_mu.mean(1))

        # if self.csc_text:
        #     text_feature_part = self.encode_text_part(text)
        #     text_token_part = self.text_proj(text_token_part)
        #     # text_token_part = F.normalize(text_token_part, dim=-1)
        #     text_token_part = text_token_part * self.textual_alpha.exp()
        #     with torch.autocast(self.textual_alpha.device.type, dtype=torch.float32, enabled = True):
        #         text_token_part = L.exp_map0(text_token_part, self.curv.exp())
        #     losses['csc_text_part'] = self.csc_loss(text_feature_part, text_token_part).mean()

        # # x_part = misc.fps(x_part.contiguous(), 1)
        # x_part = x_part.mean(1)
        # pos_mu_hyperbolic = self.pts_proj(x_part)
        # pos_mu_hyperbolic = pos_mu_hyperbolic * self.pts_alpha.exp()
        # with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32, enabled = True):
        #     pos_mu_hyperbolic = L.exp_map0(pos_mu_hyperbolic, self.curv.exp())
        # pos_mu_hyperbolic = self.mid_point(pos_mu_hyperbolic, self.curv.exp())


        # # Compute logits for contrastive loss.
        # # pts_logits = -L.pairwise_dist(x_full_hyperbolic, text_token, _curv)
        # # text_logits = -L.pairwise_dist(text_token, x_full_hyperbolic, _curv)
        # # targets = torch.arange(B, device=pts_logits.device)
        # # self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        # # _scale = self.logit_scale.exp()
        # # losses['contrastive_loss'] = 0.5 * (
        # #         nn.functional.cross_entropy(_scale * pts_logits, targets)
        # #         + nn.functional.cross_entropy(_scale * text_logits, targets)
        # # )


        # losses = self.cal_entail_loss(text_feature, text_feature_part, img_feature, x_full_hyperbolic, pos_mu_hyperbolic, losses, _curv)
        # losses = self.cal_entail_loss(text_token, text_token_part, img_token, x_full_hyperbolic, pos_mu_hyperbolic, losses, _curv)
        # losses['entailment_loss'] *= 0.001
        # losses['entailment_loss'] *= 0.1

        # losses = self.cal_nltk_loss(text, text_token, img_token, x_full_hyperbolic, losses, _curv)
        # losses['nltk'] *= 0.00001

        # losses = self.cal_contrast_with_gap_loss(text_token, img_token, x_full_hyperbolic, pos_mu_hyperbolic, losses, _curv)
        # losses['contrast_with_gap'] *= 0.0001

        # # losses['cetroid_loss'] = (L.pairwise_dist(x_full_hyperbolic, text_token, _curv).mean().norm(2) - L.pairwise_dist(img_token, x_full_hyperbolic, _curv).mean().norm(2)).norm(2)
        # centroid_inside = 1/torch.sqrt(_curv)*torch.acosh(1/_curv)
        # centroid_mid = 1/torch.sqrt(_curv)*torch.acosh(1.2/_curv)
        # centroid_outside = 1/torch.sqrt(_curv)*torch.acosh(2/_curv)
        # # centroid_inside = 1/torch.sqrt(_curv)*torch.acosh(1/_curv)
        # # centroid_mid = 1/torch.sqrt(_curv)*torch.acosh(1.05/_curv)
        # # centroid_outside = 1/torch.sqrt(_curv)*torch.acosh(1.1/_curv)
        # if torch.isnan(centroid_inside) or torch.isnan(centroid_outside) or torch.isnan(centroid_mid):
        #     print('NaN detected')
        # if torch.isinf(centroid_mid) or torch.isinf(centroid_inside) or torch.isinf(centroid_outside):
        #     print('Inf detected')
        # losses['cetroid_loss'] = ((L.refined_einstein_midpoint(text_token, _curv) - centroid_inside).norm()
        #                           + (L.refined_einstein_midpoint(x_full_hyperbolic, _curv) - centroid_mid).norm()
        #                           + (L.refined_einstein_midpoint(img_token, _curv) - centroid_outside).norm())
        # losses['cetroid_loss'] *= 0.001
        #
        # losses['orth_loss'] = sum(self.cal_orthogonal_loss(text_token, x_full_hyperbolic, img_token, _curv))
        # losses['orth_loss'] *= 0.001

        for loss_i in losses:
            if torch.isnan(losses[loss_i]):
                print(f'NaN detected in {loss_i}')

        # mar_par, data, n_points = get_children_np(pts.permute(0, 2, 1), kmin=800, kmax=1024)  # 800-1024
        # mar, pos_data, _ = get_children_np(data, starting=n_points, kmin=200, kmax=600, k=384)  # 50-600
        # pos_neighborhood, pos_center = self.pos_group_divider(pos_data.permute(0, 2, 1).contiguous())

        # pn, posn, pd, nd, t_loss, h_loss = hype_triplet_losses(x_full_hyperbolic, pos_mu_hyperbolic, hier_margin=mar, contr_margin=4, ball_dim=512)
        # losses['ht_loss'] = 0.01 * t_loss + 0.01 * h_loss
        # print(f'loss:{losses}')

        # select = torch.zeros([B, B], dtype=torch.uint8).to(img_token.device)
        # simi_pts = torch.zeros([B, B]).to(img_token.device)
        # index_sum = 0
        # for i in range(B):
        #     for j in range(B):
        #         if text[i] == text[j]:
        #             index_sum += 1
        #             if index_sum == 1:
        #                 sum_hyper = x_full_hyperbolic[i] + x_full_hyperbolic[j]
        #             else:
        #                 sum_hyper += x_full_hyperbolic[i] + x_full_hyperbolic[j]
        # sum_hyper = sum_hyper/2
        # mean = torch.mean(sum_hyper, dim=0)
        # losses['variance'] = torch.mean((sum_hyper - mean) ** 2, dim=0)
        # losses['variance'] *= 0.01
        #             select[i, j] = 1
        #         simi_pts[i, j] = L.pairwise_dist(x_full_hyperbolic[i], x_full_hyperbolic[j])+L.pairwise_dist(text_token[i], text_token[j])

        loss = sum(losses.values())
        # loss = self.awl(losses)
        # loss = self.calculate_total_loss(losses)
        return loss, [text_token, text_feature, x_full_hyperbolic, img_token, img_feature]

    def mid_point(self, x, c) -> torch.Tensor:
        """
        Compute the midpoint of points on the manifold.

        Parameters:
            x (torch.Tensor): Points on the manifold.
            w (torch.Tensor, optional): Weights for each point. Default is None.

        Returns:
            torch.Tensor: Midpoint.
        """
        ave = x.mean(dim=-2)
        denom = (-L.inner(ave, ave)).abs().clamp_min(1e-8).sqrt()
        return c.sqrt() * ave / denom.unsqueeze(-1)

    def calculate_total_loss(self, losses):
        loss_sum = 0
        for i, loss_name in enumerate(losses):
            # loss_sum += 0.5 * (torch.exp(-self.uncertainty[i]) * loss + (self.uncertainty[i]) ** 2)
            loss = losses[loss_name]
            if loss_name == 'mdm':
                loss_sum += 0.5 * (torch.exp(-self.uncertainty[0]) * loss + (self.uncertainty[0])**2)
            else:
                loss_sum += 0.5 * (torch.exp(-self.uncertainty[1]) * loss + (self.uncertainty[1])**2)
            # if i < 1:
            #     loss_sum += torch.exp(-self.uncertainty[i]) * loss + 0.5 * self.uncertainty[i]
            # else:
            #     loss_sum += 0.5 * (torch.exp(-self.uncertainty[i]) * loss + self.uncertainty[i])
        return loss_sum

    def cal_nltk_loss(self, text, text_token, img_token, x_full_hyperbolic, losses, _curv):
        B = len(text)
        select = torch.zeros([B, B]).to(text_token.device)
        for i in range(B):
            for j in range(i, B):
                if text[i] not in self.ntlk_dict:
                    text_i = wn.synset(text[i].split(',')[0].replace(' ', '_') + '.n.01')
                    self.ntlk_dict[text[i]] = text_i
                else:
                    text_i = self.ntlk_dict[text[i]]
                if text[j] not in self.ntlk_dict:
                    text_j = wn.synset(text[j].split(',')[0].replace(' ', '_') + '.n.01')
                    self.ntlk_dict[text[j]] = text_j
                else:
                    text_j = self.ntlk_dict[text[j]]
                if (text[i], text[j]) not in self.ntlk_dict:
                    select[i, j] = text_i.path_similarity(text_j)
                    self.ntlk_dict[(text[i], text[j])] = select[i, j]
                else:
                    select[i, j] = self.ntlk_dict[(text[i], text[j])]

        # similarity_text = torch.matmul(text_token, text_token.permute(1, 0))
        # similarity_img = torch.matmul(img_token, img_token.permute(1, 0))
        similarity_pts = torch.matmul(x_full_hyperbolic, x_full_hyperbolic.permute(1, 0))
        losses['nltk'] = self.csc_loss(similarity_pts, select).mean()
        # losses['nltk'] = self.csc_loss(similarity_text, select).mean()
        # losses['nltk'] += self.csc_loss(similarity_text, similarity_img).mean()
        # losses['nltk'] += self.csc_loss(similarity_text, similarity_pts).mean()
        return losses

    def cal_entail_loss(self, text_token, text_token_part, img_token, x_full_hyperbolic, pos_mu_hyperbolic, losses, _curv):
        _aperture_pts = L.half_aperture(x_full_hyperbolic, _curv)
        _aperture_txt = L.half_aperture(text_token, _curv)
        _aperture_txt_part = L.half_aperture(text_token_part, _curv)
        # # 0
        # _angle_txt_img = L.exterior_angle(text_token, img_token, _curv)
        # losses['entailment_loss'] = torch.clamp(_angle_txt_img - _aperture_txt, min=0).mean()
        # 1
        _angle_txt_pts = L.exterior_angle(text_token, x_full_hyperbolic, _curv)
        losses['entailment_loss'] = torch.clamp(_angle_txt_pts - _aperture_txt, min=0).mean()
        # 2
        _angle_pts_img = L.exterior_angle(x_full_hyperbolic, img_token, _curv)
        losses['entailment_loss'] += torch.clamp(_angle_pts_img - _aperture_pts, min=0).mean() # svm acc = 91.1264 / 87.8039 / 89.7893
        # 3
        _angle_pts_part = L.exterior_angle(x_full_hyperbolic, pos_mu_hyperbolic, _curv)
        losses['entailment_loss'] += torch.clamp(_angle_pts_part - _aperture_pts, min=0).mean()
        # 4
        # _angle_txt_pts_part = L.exterior_angle(text_token_part, pos_mu_hyperbolic, _curv)
        # losses['entailment_loss'] += torch.clamp(_angle_txt_pts_part - _aperture_txt_part, min=0).mean()
        # 5
        # _anglet_txt_part = L.exterior_angle(text_token, text_token_part, _curv)
        # losses['entailment_loss'] += torch.clamp(_anglet_txt_part - _aperture_txt, min=0).mean()

        # 20241013: exterior angle_txt_pts > .._pts_img --> max(pts_img - txt_pts < 0, 0)=0
        # losses['entailment_loss'] += torch.clamp(_angle_pts_img - _angle_txt_pts, min=0).mean()
        # https://github.com/facebookresearch/poincare-embeddings/blob/main/hype/energy_function.py#L56
        # https://github.com/facebookresearch/poincare-embeddings/blob/b1364a1c19a6ab28256c8c11c1b1064a1038b762/hype/hypernymy_eval.py#L56
        return losses

    def cal_contrast_with_gap_loss(self, text_token, img_token, x_full_hyperbolic, pos_mu_hyperbolic, losses, _curv):
        # T-->point clouds
        losses['contrast_with_gap'] = self.exterior_angle_contra_head(text_token, x_full_hyperbolic, _curv)
        # point clouds-->image
        losses['contrast_with_gap'] += self.exterior_angle_contra_head(x_full_hyperbolic, img_token, _curv)
        return losses

    # @torch.no_grad()
    def cal_orthogonal_loss(self, text_token, x_full_hyperbolic, img_token, _curv):
        import torch.nn.functional as F
        diff_text_token = text_token - L.refined_einstein_midpoint(text_token, _curv)
        diff_pts_full_token = x_full_hyperbolic - L.refined_einstein_midpoint(x_full_hyperbolic, _curv)
        diff_img_token = img_token - L.refined_einstein_midpoint(img_token, _curv)
        # diff_text_token = L.refined_einstein_midpoint(diff_text_token, _curv)
        # diff_pts_full_token = L.refined_einstein_midpoint(diff_pts_full_token, _curv)
        # diff_img_token = L.refined_einstein_midpoint(diff_img_token, _curv)

        text2pts_token = L.refined_einstein_midpoint(x_full_hyperbolic, _curv) - L.refined_einstein_midpoint(text_token, _curv)
        pts2img_token = L.refined_einstein_midpoint(x_full_hyperbolic, _curv) - L.refined_einstein_midpoint(img_token, _curv)
        img2text_token = L.refined_einstein_midpoint(text_token, _curv) - L.refined_einstein_midpoint(img_token, _curv)

        orth_loss_txt2pts = F.cosine_similarity(diff_text_token, text2pts_token.expand_as(diff_text_token), dim=1).norm()
        orth_loss_pts2img = F.cosine_similarity(diff_pts_full_token, pts2img_token.expand_as(diff_pts_full_token), dim=1).norm()
        orth_loss_img2txt = F.cosine_similarity(diff_img_token, img2text_token.expand_as(diff_img_token), dim=1).norm()
        # losses['orth_loss'] = (diff_text_token @ text2pts_token).norm()
        # losses['orth_loss'] += (diff_pts_full_token @ pts2img_token).norm()
        # losses['orth_loss'] += (diff_img_token @ img2text_token).norm()

        return orth_loss_txt2pts, orth_loss_pts2img, orth_loss_img2txt


# def hype_triplet_losses(parent_mu, pos_child_mu, hier_margin=0.2, contr_margin=4, ball_dim=512, one_child=False,
#                         opposite_hier=False):
#     ball = PoincareBall(c=1.0, dim=ball_dim)
#     # part1
#     parent_norm = torch.norm(parent_mu, dim=1)  # dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
#     pos_norm = torch.norm(pos_child_mu, dim=1)
#
#     # part2
#     if one_child:
#         neg_child_mu = torch.flip(parent_mu, [0])
#     else:
#         neg_child_mu = torch.flip(pos_child_mu, [0])
#
#     # distance_positive = ball.dist(parent_mu, pos_child_mu)  # , keepdim=True).pow(2).sum(1)
#     # distance_negative = ball.dist(parent_mu, neg_child_mu)  # , keepdim=True).pow(2).sum(1)
#     distance_positive = L.pairwise_dist(parent_mu, pos_child_mu)  # , keepdim=True).pow(2).sum(1)
#     distance_negative = L.pairwise_dist(parent_mu, neg_child_mu)  # , keepdim=True).pow(2).sum(1)
#     distance = distance_positive - distance_negative + contr_margin
#     triplet = torch.mean(
#         torch.max(distance, torch.zeros_like(distance)))  # F.relu(distance_positive - distance_negative + 0.5 )
#
#     # part3
#     par_norm = ball.dist0(parent_mu)
#     p_norm = ball.dist0(pos_child_mu)  # .pow(2).sum(1)
#     if opposite_hier:
#         norm = par_norm - p_norm + hier_margin
#     else:
#         norm = -par_norm + p_norm + hier_margin  # Standard version wich improves the baseline
#     hierarch = torch.mean(torch.max(norm, torch.zeros_like(norm)))  # F.relu(-par_norm+p_norm + margin)
#
#     return parent_norm.mean(), pos_norm.mean(), distance_positive.mean(), distance_negative.mean(), triplet, hierarch


def p2k(x, c):
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


def k2p(x, c):
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom

def lorenz_factor(x, *, c=1.0, dim=-1, keepdim=False):
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
        Lorenz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def poincare_mean(x, dim=0, c=1.0):
    print(x)
    x = p2k(x, c)
    lamb = lorenz_factor(x, c=c, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(
        lamb, dim=dim, keepdim=True
    )
    mean = k2p(mean, c)
    return mean.squeeze(dim)

'''
@MODELS.register_module()
class ReCon_hyperXXX(nn.Module):
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
                    with torch.autocast(self.visual_alpha.device.type, dtype=torch.float32):
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
                    with torch.autocast(self.textual_alpha.device.type, dtype=torch.float32):
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
                losses['cetroid_loss_1'] = (self.centroid_weight *
                                            (einstein_midpoint_regularization(text_feature, img_feature, _curv)
                                             +einstein_midpoint_regularization(text_feature, x_full, _curv)
                                             )
                                            )
            if loss == None:
                loss = sum(losses.values())
            else:
                loss += sum(losses.values())
                if iter_innerloop == 2:
                    _angle_pts = L.exterior_angle(parent_mu, child_mu, _curv)
                    _aperture_pts = L.half_aperture(parent_mu, _curv)
                    losses['h_t_loss'] = self.entail_weight * torch.clamp(_angle_pts - _aperture_pts, min=0).mean()
                    losses['cetroid_loss_2'] = self.centroid_weight * hyperbolic_centroid_loss(parent_mu, child_mu)
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
        
'''