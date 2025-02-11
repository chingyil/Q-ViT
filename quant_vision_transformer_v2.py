import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from quant_vision_transformer import Q_PatchEmbed, Q_Attention, Q_Mlp
from Quant import LinearQ, ActQ
from timm.models.layers.weight_init import trunc_normal_
import numpy as np
from timm.models.layers.drop import DropPath

class Q_Block(nn.Module):

    def __init__(self, nbits, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Q_Attention(nbits, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        print("drop_path = ", drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Q_Mlp(nbits=nbits, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x0):
        x1 = x0 + self.drop_path(self.attn(self.norm1(x0)))
        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        print(np.percentile(x0, (0, 5, 95, 100)), x0.std())
        # import pdb; pdb.set_trace()
        return x2

class lowbit_VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, nbits, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=Q_PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            nbits: nbits
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # import pdb; pdb.set_trace()
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            nbits=nbits, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Q_Block(
                nbits=nbits, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = LinearQ(self.num_features, num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
        # nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = LinearQ(self.embed_dim, self.num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
        import pdb; pdb.set_trace()

    def forward_features(self, x0):
        x1 = self.patch_embed(x0)
        assert not np.isnan(x1).any()
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x2 = torch.cat((cls_token, self.dist_token.expand(x1.shape[0], -1, -1), x1), dim=1)
        assert not np.isnan(x2).any()
        x3 = self.pos_drop(x2 + self.pos_embed)
        assert not np.isnan(x3).any()
        x4 = self.blocks(x3)
        x5 = self.norm(x4)
        # import pdb; pdb.set_trace()
        return x5[:, 0], x5[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        return (x + x_dist) / 2
