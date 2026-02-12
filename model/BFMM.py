# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from thop import profile
from torch.nn import init as init
import tensorly as tl
tl.set_backend('pytorch')


class frequency_selection(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, window_size=None, bias=False):
        super().__init__()
        self.act_fft = act_method()
        self.window_size = window_size
        hid_dim = dim * dw

        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))

        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))

        self.bias = bias
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)

        y = rearrange(y, 'b c h w -> b h w c')   # (B,H,W',C)
        y = y @ weight1                         # (B,H,W',hid_dim)

        y = torch.cat([y.real, y.imag], dim=1)  # (B,2H,W',hid_dim) 注意：这行维度逻辑其实有点怪，见下方说明
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)

        y = torch.complex(y_real, y_imag)
        y = y @ weight2                         # (B,H,W',C)
        y = rearrange(y, 'b h w c -> b c h w')   # (B,C,H,W')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        return y




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand,dropout=attn_drop_rate, **kwargs)
        self.fft_branch0 = frequency_selection(dim=hidden_dim)

        # self.self_attention = UV

        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        fft0 = self.fft_branch0(input.permute(0, 3, 1, 2).contiguous())
        fft0 = fft0.permute(0, 2, 3, 1).contiguous()

        x = self.ln_1(input)  # B H W C
        x = input*self.skip_scale + fft0 + self.drop_path(self.self_attention(x))
        # x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        # print(000, x.size())
        # fft = x.permute(0, 3, 1, 2).contiguous()
        # fft = self.fft_branch(fft)
        # fft = fft.permute(0, 2, 3, 1).contiguous()
        # # x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # x = x*self.skip_scale2 + fft + self.hybridgate(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()  # B H W C

        x = x.view(B, -1, C).contiguous()
        return x


class FMJM(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



class KRIM(nn.Module):
    def __init__(self, num_direction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)                  # -> (B,C,1,1)
        self.fc = nn.Conv2d(num_direction, num_direction*2, 1, bias=False)  # -> (B,2C,1,1)
        self.bn = nn.BatchNorm1d(num_direction*2)               # 因为后面要压成 (B,2C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B,C,H,W) = (1,64,64,64)
        B, C, H, W = x.size()
        s = self.avgpool(x)                 # (B,C,1,1)
        s = self.fc(s)                      # (B,2C,1,1)
        s = s.view(B, 2*C)                  # (B,2C) 例如 (1,128)s
        s = self.sigmoid(self.bn(s))        # (B,2C)

        v1, v2 = torch.chunk(s, 2, dim=1)   # v1,v2: (B,C) 各自是(1,64)

        attn_64x64 = v1.unsqueeze(2) * v2.unsqueeze(1)  # 或 torch.einsum('bi,bj->bij', v1, v2)

        return attn_64x64


class TCFM(nn.Module):
    def __init__(self, c_in, w_in, h_in):
        super(TCFM, self).__init__()
        self.w_in = w_in
        self.h_in = h_in

        self.msa1 = FMJM(
            dim=c_in,
            input_resolution=(w_in, h_in),
            depth=2,
            d_state = 16,
            mlp_ratio=2,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            is_light_sr = False)

        self.conv_c = nn.Sequential(
            nn.Conv2d(c_in, c_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(w_in, w_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(h_in, h_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(c_in * 3, c_in, (3, 3), padding=1),
        )

        self.layernorm_c0 = nn.LayerNorm(c_in)
        self.layernorm_w0 = nn.LayerNorm(w_in)
        self.layernorm_h0 = nn.LayerNorm(h_in)

        self.layernorm1 = nn.LayerNorm(c_in)
        self.layernorm2 = nn.LayerNorm(c_in)

        self.layernorm_c = nn.LayerNorm(c_in)
        self.layernorm_w = nn.LayerNorm(w_in)
        self.layernorm_h = nn.LayerNorm(h_in)

        self.krim_c1 = KRIM(c_in)
        self.krim_h1 = KRIM(h_in)
        self.krim_w1 = KRIM(w_in)
    def forward(self, x_in): # (1,64,64,64)
        b_in, c_in, w_in, h_in = x_in.shape # (16,64,64,64)

        x_c = x_in  # b c w h # (16,64,64,64)
        x_w = x_in.permute(0, 2, 1, 3)  # b w c h # (16,64,64,64)
        x_h = x_in.permute(0, 3, 2, 1)  # b h w c # (16,64,64,64)
        x_c = x_c.permute(0, 3, 2, 1)   # b h w c # (16,64,64,64)
        # 所有msa的输出形状与输入一致，一个主干编码，一个位置编码
        x_c = self.layernorm_c0(x_c) # (16,64,64,64)
        
        # print("msa_c")
        x_h = x_h.permute(0, 2, 3, 1)  # b h w  c -->  b, w, c, h # (16,64,64,64) # (1,64,64,64)
        x_h = self.layernorm_h0(x_h) # (16,64,64,64)
        
        x_w = x_w.permute(0, 2, 3, 1)  # b w c h -->  b, c, h ,w  # (16,64,64,64)
        x_w = self.layernorm_w0(x_w) # (16,64,64,64)
    
        out_c = self.conv_c(x_c.permute(0, 3, 1, 2))  # (16,64,64,64) # out: b h w c --> b c h w 
        out_h = self.conv_h(x_h.permute(0, 3, 1, 2))  # (16,64,64,64) out: b w c h  -> b h w c
        out_w = self.conv_w(x_w.permute(0, 3, 1, 2))  # (16,64,64,64) out: b c h w --> b w c h
        

        vector_c = self.krim_c1(out_c) 
        vector_h = self.krim_h1(out_h)
        vector_w = self.krim_w1(out_w)

        xx = rearrange(x_in, "b c w h -> b h w c", h=self.h_in, w=self.w_in).contiguous() #(64,64,64)
        xx = self.layernorm1(xx) # (16,64,64,64) b c w h
        xx = rearrange(xx, "b h w c -> b (h w) c", h=self.h_in, w=self.w_in).contiguous() # (16,4096,64)
        xx= self.msa1(xx, (self.w_in, self.h_in)) # (16,4096,64) 这应该是FMJM模块被我删掉了
        xx = rearrange(xx, "b (h w) c -> b c w h ", h=self.h_in, w=self.w_in).contiguous() #(16,64,64,64)

        results = []
        for i in range(b_in):
            res = tl.tucker_to_tensor((xx[i], [vector_c[i], vector_w[i], vector_h[i]]))
            results.append(res)
        result = torch.stack(results)
        
        result = x_in + result

        return xx


class BTDR(nn.Module):
    def __init__(self, c_in, w_in, h_in):
        super(BTDR, self).__init__()
        self.TCFM_num = 3
        self.TCFM1 = TCFM(c_in, w_in, h_in)
        self.TCFM2 = TCFM(c_in, w_in, h_in)
        self.TCFM3 = TCFM(c_in, w_in, h_in)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in * self.TCFM_num, c_in, (3, 3), padding=1),
        )

    def forward(self, x):
        o1 = self.TCFM1(x)
        o2 = self.TCFM2(x - o1)
        o3 = self.TCFM3(x - o1)
        o_all = torch.cat((o1, o2, o3), dim=1)
        out = self.conv(o_all)
        return out


class CA(nn.Module):
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sig(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class MDCA(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 3), use_eca=True):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, padding=d, dilation=d, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ) for d in dilations
        ])
        self.point = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.eca = CA(channels) if use_eca else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        y = self.point(y)
        y = self.eca(y)
        return self.act(x + y)


class LRRB(nn.Module):
    def __init__(self, cp_in_c, w_in, h_in):
        super(LRRB, self).__init__()

        self.BTDR = BTDR(cp_in_c, w_in, h_in)
        self.MDCA = MDCA(channels=cp_in_c, dilations=(1, 2, 3), use_eca=True)

        
        self.conv = nn.Conv2d(cp_in_c*2, cp_in_c, (1, 1), (1, 1), bias=False)

    def forward(self, x):
        F = self.BTDR(x)
        E = self.MDCA(F)
        out = self.conv(torch.cat([E, x], dim=1))
        return out


class BFMM(nn.Module):
    def __init__(self, opt):
        super(BFMM, self).__init__()
        self.sf = opt.sf
        c_in=opt.hschannel+opt.mschannel
        w_in=64 
        h_in=64
        cp_in_c=64
        c_out=opt.hschannel
        self.stage = 2

        self.conv0 = nn.Conv2d(c_in, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.LRRB1 = LRRB(cp_in_c, w_in, h_in)
        self.LRRB2 = LRRB(cp_in_c, w_in, h_in)

        self.conv = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
        )
        
        self.conv00 = nn.Conv2d(cp_in_c, c_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=self.sf, mode='bicubic', align_corners=False)
        xx = torch.cat((x, y),dim=1)
        
        x0 = self.conv0(xx)
        
        fea1 = self.LRRB1(x0)
        fea2 = self.LRRB2(fea1)
        
        out = self.conv(fea2 + x0)
        out = self.conv00(out) + x 
        return out