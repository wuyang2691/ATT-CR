
from math import prod
from typing import Tuple

import numpy as np
import torch
from timm.models.layers import to_2tuple

from torch import nn
import math
from torch.nn import functional as F
from einops import rearrange as rearrange
from abc import ABC
from .tools import *


# select attention 
class FeaFuse(nn.Module):
    def __init__(self, in_channels, num_fea=2, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.num = num_fea
        for i in range(num_fea):
            block = nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, padding=0, bias=False)
            setattr(self, 'trans' + str(i), block)
        self.softmax= nn.Softmax(dim=1)

    def forward(self, batch):
        bs = batch[0].size(0)
        ch = batch[0].size(1)
        batch = torch.cat(batch, dim=1).view(bs, self.num,  ch, batch[0].size(2), batch[0].size(3))
        temp = torch.sum(batch, dim=1)
        temp = self.avg_pool(temp)
        temp = self.se(temp)

        attn0 = self.trans0(temp)
        attn1 = self.trans1(temp)
       

        battn = torch.cat([attn0, attn1], dim=1).view(bs, self.num, ch, 1, 1)
        battn = self.softmax(battn)
        feature = torch.sum(battn * batch, dim=1)
        return feature

class Glolbal_Attention(nn.Module):
    def __init__(self, in_ch=256, num_head=4, bias=False):
        super().__init__()
        self.head = num_head
        self.query = nn.Sequential(
            # nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            # nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1,groups=in_ch,bias=bias),
            nn.Softplus()
        )

        self.key = nn.Sequential(
            # nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            # nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1,groups=in_ch,bias=bias),
            nn.Softplus()
            # nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        # self.value = nn.Sequential(
        #     nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0,bias=bias),
        #     nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1,groups=in_ch,bias=bias),
        # )#nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        # if in_ch < 48:
        #     pass
        # else:
        #     self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        # self.norm = nn.InstanceNorm2d(num_features=in_ch)

    def forward(self, x):
        """
        x: b * c * h * w
        """

        
        # import pdb 
        # pdb.set_trace()
        n, Ba, Ca, He, We = x.size()
        q, k, v = x[0], x[1], x[2] 
        q = self.query(q)
        k = self.key(k)
        # q = self.query(x)
        # k = self.key(x)
        # v = self.value(x)
        num_per_head = Ca // self.head

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head) # B * head * c * N
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head) # B * head * c * N
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)  # B * head * c * N
        kv = torch.matmul(k, v.transpose(-2, -1))
       
        z = torch.einsum('bhcn,bhc -> bhn', q, k.sum(dim=-1)).contiguous() / math.sqrt(num_per_head)
        z = 1.0 / (z + He * We)  # b h n
        out = torch.einsum('bhcn, bhcd-> bhdn', q, kv).contiguous()
        out = out / math.sqrt(num_per_head)  # b h c n
        out = out * z.unsqueeze(2)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=He)  # b,C,h,w
       
        #
        #     out = self.output_linear(out)
       
        return out  # b,c,h,w


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, args=None):
        m = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
        ]
        # if args.separable_conv_act:
        #     m.append(nn.GELU())
        m.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias))
        super(SeparableConv, self).__init__(*m)


class QKVProjection(nn.Module):
    def __init__(self, dim, qkv_bias):
        super(QKVProjection, self).__init__()
    
        self.body = SeparableConv(dim, dim * 3, 3, 1, qkv_bias)

    def forward(self, x, x_size):
        # if self.proj_type == "separable_conv":
      
        x = self.body(x)
        # if self.proj_type == "separable_conv":
        x = bchw_to_blc(x)
        return x
###############window attention#####################
class Attention(ABC, nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def attn(self, q, k, v, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
       
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        # attn = attn_transform(attn, table, index, mask)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class WindowAttention(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads=3,
        attn_drop=0.0,
        args=None,
    ):

        super(WindowAttention, self).__init__()
       
        self.window_size = [window_size, window_size] 
      
        self.num_heads = num_heads
      
        self.qkv =  QKVProjection(dim, qkv_bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_size):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        qkv = self.qkv(x, x_size)  # b,l,c
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)
        
        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # nW*B, H, wh*ww, dim

        # attention
        x = self.attn(q, k, v)

        # merge windows
        x = x.view(-1, *self.window_size, C//3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C
       
        x = bhwc_to_bchw(x)
        # x = x.view(B, L, C//3)

        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}" #, shift_size={self.shift_size}, "
            # f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        pass

class CPB_MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, channels=512):
        m = [
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        ]
        super(CPB_MLP, self).__init__(*m)

class Attention_2(ABC, nn.Module):
    def __init__(self):
        super(Attention_2, self).__init__()

    def attn(self, q, k, v, attn_transform, table, index, mask, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
        scale = head_dim ** -0.5
        if self.euclidean_dist:
            # print("use euclidean distance")
            attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        else:
            attn = (q @ k.transpose(-2, -1)) *scale  #F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, table, index, mask)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x

class AffineTransform(nn.Module):
    r"""Affine transformation of the attention map.
    The window could be a square window or a stripe window. Supports attention between different window sizes
    """

    def __init__(self, num_heads):
        super(AffineTransform, self).__init__()
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)

    def forward(self, attn, relative_coords_table, relative_position_index, mask):
        B_, H, N1, N2 = attn.shape
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        bias_table = self.cpb_mlp(relative_coords_table)  # 2*Wh-1, 2*Ww-1, num_heads
        bias_table = bias_table.view(-1, H)

        bias = bias_table[relative_position_index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, H, N1, N2) + mask
            attn = attn.view(-1, H, N1, N2)

        return attn

class WindowAttention_v2(Attention_2):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        window_size,
        num_heads,
        window_shift=False,
        attn_drop=0.0,
        pretrained_window_size=[0, 0],
        args=None,
    ):

        super(WindowAttention_v2, self).__init__()
        # self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.window_shift = window_shift
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = False

        self.attn_transform = AffineTransform(num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        
      
    
    

    def forward(self, qkv, x_size, table, index, mask):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        n, B, c, H, W = qkv.shape
        C = n*c
        qkv = qkv.permute(1,0,2,3,4).contiguous().view(B, C, H, W).permute(0,2,3,1).contiguous() # b, 3c, H,W
        
        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C).contiguous()  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # nW*B, H, wh*ww, dim

        # attention
        x = self.attn(q, k, v, self.attn_transform, table, index, mask)

        # merge windows
        x = x.view(-1, *self.window_size, C // 3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # x = x.view(B, L, C // 3)
        x = bhwc_to_bchw(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        pass