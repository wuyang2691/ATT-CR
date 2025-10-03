## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

import math
from .causal_linear_attention import  CausalLinearAttention
from .masking import TriangularCausalMask, FullMask
# from triton_rms_norm import TritonRMSNorm2dFunc
from ..plot_feature import *  #

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MS_Linear_Attention(nn.Module):
    def __init__(self, in_ch=256, num_head=4, scales=[3,], bias=False):
        super().__init__()
        self.head = num_head
        self.scales = scales
        self.in_ch = in_ch
       
        self.dim = in_ch // self.head
       
        self.query = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
       # multi-scale token
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=in_ch, out_channels=2*in_ch, kernel_size=1, padding=0,bias=bias),
                    nn.Conv2d(
                        2 * in_ch,
                        2 * in_ch,
                        scale,
                        padding=get_same_padding(scale),
                        groups=2 * in_ch,
                        bias=False,
                    ) if scale > 1 else nn.Identity()
                )
                for scale in scales
            ]
        )
       
        if  len(scales)>1:
            self.proj = nn.Linear(len(scales)*in_ch, in_ch) 
        self.eps = 1e-6
        self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)

        
        self.causal_att_down_tri = CausalLinearAttention(in_ch//self.head)
     
     
        
            
        self.gate = nn.Sequential(nn.Conv2d(in_ch, in_ch // 4, 1, 1, 0),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_ch // 4, in_ch // 4, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_ch // 4, in_ch, 1, 1, 0))
        
       



      
    def forward(self, x):
        """
        x: b * c * h * w
        """  

        Ba, Ca, He, We = x.size()
       
        num_per_head = Ca // self.head
        N = He*We

        q = self.query(x)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head, h=He, w=We)
        multi_scale_kv = []
        for op in self.aggreg:
            kv_s = op(x) # b, c, h, w
            multi_scale_kv.append(kv_s)
        
        n_scale= len(self.scales)
        multi_scale_kv_hw = []
        multi_scale_kv_wh = []
        for i in range(n_scale):
            kv_s_i = multi_scale_kv[i] # 对kv 进行不同尺度的卷积操作
            kv_s_i= kv_s_i.view(Ba, 2, Ca, N) # split为 K, V
            kv_s_i = rearrange(kv_s_i, 'b n (head c) (h w) -> b n head c (h w)', head=self.head, h=He, w=We) # b 2 head c (h w)
            kv_s_i_hw = kv_s_i[:, :, 0:(self.head//2)] # b 2 head//2 c (h w) 一半head作为hw
            kv_s_i_wh = kv_s_i[:, :, (self.head//2):].flip(4).contiguous() # b 2 head//2 c (wh) 一半head 为wh
            multi_scale_kv_hw.append(kv_s_i_hw) #将每个尺度hw保存在列表
            multi_scale_kv_wh.append(kv_s_i_wh)
         
        multi_scale_kv_hw = torch.cat(multi_scale_kv_hw, dim=2)# b, 2, head//2 *(n_scale), c (h w) #将每个尺度hw进行拼接
        multi_scale_kv_wh = torch.cat(multi_scale_kv_wh, dim=2)# b, 2, head//2 *(n_scale), c (h w)#将每个尺度wh进行拼接
        multi_scale_kv_all = torch.cat([multi_scale_kv_hw, multi_scale_kv_wh], dim=2) # b, 2, head *(n_scale), c (h w)
        multi_scale_kv_all = multi_scale_kv_all.permute(1, 0, 4, 2, 3).contiguous()# 2,  B,  (h w), head *(n_scale) c
        q_hw = []
        q_wh = []
        
        for i in range(n_scale):
            q_hw_i = q[:, 0:(self.head//2)]
            q_wh_i = q[:, (self.head//2):].flip(3).contiguous()
            q_hw.append(q_hw_i)
            q_wh.append(q_wh_i)
        q_hw = torch.cat(q_hw, dim=1) # b (head//2 * n_scale) c (h w)
        q_wh = torch.cat(q_wh, dim=1) # b (head//2 * n_scale) c (h w)

       

        q = torch.cat([q_hw, q_wh], dim=1).permute(0, 3, 1, 2).contiguous() # B  N  head c 
        k = multi_scale_kv_all[0]
        v = multi_scale_kv_all[1] # B  N  head c 
       

        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        N = He * We
        mask_tri = TriangularCausalMask(N, device="cuda")
        query_mask = FullMask(int(Ba), int(N), device="cuda")
        key_mask = FullMask(int(Ba), int(N), device="cuda")


        # 下三角矩阵
        all_head = n_scale * self.head
        

        v_down_tri = self.causal_att_down_tri(q, k, v, mask_tri, query_mask, key_mask)
        
        v_flip = v_down_tri[:, :, (all_head//2):, :].flip(1).contiguous() #(B  N  head//2  c )
        v_hw = v_down_tri[:, :, 0:(all_head//2)]
        
        #对每个尺度的头进行分别处理
        v_new = []
        for i in range(n_scale):
            temp_hw= v_hw[:, :, (i*self.head//2):((i+1)*self.head//2)]
            temp_wh = v_flip[:, :, (i*self.head//2):((i+1)*self.head//2)]
            temp_scale_i = torch.cat([temp_hw, temp_wh], dim=2).view( Ba, N, -1) #(B  N  head*c ) scale i    
            v_new.append(temp_scale_i)
        
       
        v_new = torch.cat(v_new, dim=-1)
        if n_scale>1:
            v_new = self.proj(v_new) #(B  N  C )
       
        
       
        out = v_new.transpose(1,2).view(Ba, Ca, He ,We ).contiguous()#(B   C H W )
       
        #  add GATE
        gate = self.gate(x) #
        
        out = gate * out 

        out = self.output_linear(out)

        return out  # b,c,h,w



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, scales):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.gloabl_attn = MS_Linear_Attention(dim, num_heads, scales)
    

    def forward(self, x):
        x = x + self.gloabl_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        # mod_pad_h, mod_pad_w = 0, 0
        # _, _, h, w = x.size()
        # if h % 2 != 0:
        #     mod_pad_h =  h % 2
        # if w % 2 != 0:
        #     mod_pad_w = w % 2
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################


class ATT_CR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [1, 2, 8, 2, 1], 
        num_refinement_blocks = 4,
        heads = [2,2,8,2,2],
        scales = [[3],[3], [3], [3], [3], [3]],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(ATT_CR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], scales= scales[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], scales= scales[1],ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], scales= scales[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

       
        
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**2), int(dim*2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2), num_heads=heads[3], scales= scales[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[4], scales= scales[4], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[4])])

        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
                        
        inp_dec_level3 = self.up3_2(out_enc_level3)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 


        inp_dec_level1 = self.up2_1(out_dec_level3)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)


   
        if inp_img.shape[1]>13: # deal with SAR data
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:, 2:, :,:]
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img



        return out_dec_level1