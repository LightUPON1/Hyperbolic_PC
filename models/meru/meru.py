# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re
from collections import OrderedDict
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import timm
# from config import LazyCall, LazyFactory, LazyConfig
# from models import MERU, CLIPBaseline
# from tokenizer import Tokenizer
# from config import LazyConfig, LazyFactory

from config import LazyCall
from util_meru_models import MERU
from tokenizer import Tokenizer
from config import LazyConfig, LazyFactory

class _TransformerBlock(nn.Module):
    """
    Single transformer block comprising multi-head self-attention and MLP. Both
    modules are preceeding by layer normalization. This module is same as PyTorch
    builtin module `TransformerEncoderLayer` with arguments as
    (`norm_first=True, dropout=0, activation="gelu"`).

    We adapt this module from CLIP to easily load checkpoints of CLIP and other
    works that build on CLIP's code. Reference: https://github.com/openai/clip
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, lx, need_weights=False, attn_mask=attn_mask)[0]
        x = x + ax
        x = x + self.mlp(self.ln_2(x))
        return x


class TextTransformer(nn.Module):
    """
    Text encoder using multiple layers of transformer encoder blocks. It accepts
    tokenized text sequences, passes them through word/position embedding layers
    and further processes them through transformer layers.

    All transformer blocks are unidirectional "Pre-LN" variants by default:
    LayerNorm is placed before attention/MLP layers inside the residual block,
    and future positions are masked while computing self-attention.
    """

    def __init__(
        self,
        arch: str,
        vocab_size: int,
        context_length: int,
        grad_checkpointing: bool = False,
    ):
        """
        Args:
            arch: Architecture config for transformer, describing layers, width,
                and number of attention heads. For example, `L12_W512_A8` has 1
                layer, 512 width, 8 heads. Width of MLP will always be `4 * W`,
                per transformer paper. `A` is optional and will default to
                (`A = H/64`) per transformer paper.
            vocab_size: Number of tokens in the output vocabulary.
            context_length: Maximum length of input captions; this is used to
                create a fixed positional embedding lookup table.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing

        # Parse architecture str: layers, width, heads, feed-forward size.
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))

        # Find heads in architecture else use (H // 64) per (Vaswani et al.)
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64

        # Input sequences in forward pass will be right padded with zeroes.
        # `nn.Embedding` has a `padding_idx` argument to set their embedding as
        # zero. However, since the blocks are uni-directional, they will never
        # receive gradients for padded positions.
        self.token_embed = nn.Embedding(vocab_size, self.width)
        self.posit_embed = nn.Parameter(torch.empty(context_length, self.width))

        # Make a sequential module of transformer encoder blocks.
        _resblocks = [
            _TransformerBlock(self.width, self.heads) for _ in range(self.layers)
        ]
        self.resblocks = nn.ModuleList(_resblocks)
        self.ln_final = nn.LayerNorm(self.width)

        # Generate a unidirectional mask for self-attention. As per PyTorch API,
        # masked positions are set to `-inf`.
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())

        # Initialize all modules like CLIP:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.posit_embed.data, std=0.01)

        out_proj_std = (2 * self.width * self.layers) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=self.width**-0.5)
            nn.init.normal_(block.attn.out_proj.weight, std=out_proj_std)
            nn.init.normal_(block.mlp[0].weight, std=(2 * self.width) ** -0.5)
            nn.init.normal_(block.mlp[2].weight, std=out_proj_std)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Obtain features of input text tokens by passing them through transformer
        blocks. All self-attention layers only attend to past token (left side).
        """

        max_len = text_tokens.shape[-1]
        _posit_embed = self.posit_embed[:max_len, :]
        _attn_mask = self.attn_mask[:max_len, :max_len]

        # shape: (batch_size, context_length, width)
        token_embeddings = self.token_embed(text_tokens) + _posit_embed

        # Forward pass through transformer, optionally with grad checkpointing.
        textual_features = token_embeddings
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                # shape: (context_length, batch_size, width)
                textual_features = checkpoint(block, textual_features, _attn_mask)
            else:
                textual_features = block(textual_features, _attn_mask)

        textual_features = self.ln_final(textual_features)
        return textual_features

def build_timm_vit(
    arch: str,
    global_pool: str = "token",
    use_sincos2d_pos: bool = True,
    grad_checkpointing: bool = False,
):
    """
    Build a Vision Transformer (ViT) image encoder from timm.

    Args:
        global_pool: How to perform global pooling after final layer? If this is
            `token` (default), we use the [cls] token. For `avg` we perform
            global average pooling like ConvNets.
        use_sincos2d_pos: Use a fixed 2D sine-cosine position embedding. If this
            is False; position embeddings are randomly initialized (and learned).
    """

    _supported = timm.list_models("vit_*")
    if arch not in _supported:
        raise ValueError(f"{arch} is not a supported ViT, choose: {_supported}")

    model = timm.create_model(
        arch,
        # `num_classes = 0` does not create the final classification head.
        num_classes=0,
        global_pool=global_pool,
        #
        # Do not use [CLS] token for models that use global average pooling.
        class_token=global_pool == "token",
        #
        # Use LayerNorm with default `eps = 1e-5` (timm uses 1e-6, not sure why)
        norm_layer=nn.LayerNorm,
    )
    model.set_grad_checkpointing(grad_checkpointing)

    # Set `width` attribute to access in model.
    model.width = model.embed_dim

    # ------------------------------------------------------------------------
    # 2D sine-cosine embedding for Vision Transformers. This implementation
    # is adapted from MoCo-v3 (https://github.com/facebookresearch/moco-v3) and
    # it produces exactly same embeddings.
    # ------------------------------------------------------------------------
    if use_sincos2d_pos:
        h, w = model.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)

        assert (
            model.embed_dim % 4 == 0
        ), "ViT embed_dim must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = model.embed_dim // 4

        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (10000.0**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        if global_pool == "token":
            pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
            pos_emb = torch.cat([pe_token, pos_emb], dim=1)

        # Override position embedding.
        model.pos_embed.data.copy_(pos_emb)
        model.pos_embed.requires_grad = False

    return model

model_MERU = LazyCall(MERU)(
    visual=LazyCall(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=LazyCall(TextTransformer)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
)

#
# tokenizer = Tokenizer()
train_config='/home/ubuntu/CodeSpace/Hyperbolic_PC/models/meru/config_train_meru_vit_b.py'
checkpoint_path = '/home/ubuntu/CodeSpace/meru-main/checkpoint/meru_vit_b.pth'

def load_meru(train_config = train_config, checkpoint_path = checkpoint_path):
    # print(f'laod meru checkpoint')
    device = (
        torch.cuda.current_device()
    )
    _C_TRAIN = LazyConfig.load(train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    for p in model.parameters():
        p.requires_grad = False
    # for p in model.visual.parameters():
    #     p.requires_grad = False
    # for p in model.textual.parameters():
    #     p.requires_grad = False
    return model
    # return model.visual, model.textual, model.encode_image, model.encode_text


