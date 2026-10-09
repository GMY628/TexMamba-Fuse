"""Mamba feature encoders, fusion decoders, and text-driven enhancement.

Original source header: Code Implementation of the MambaIR Model.

Requires PyTorch, einops, timm, and the original csm_triton/csms6s modules
with their scan backends. Public class names, constructor signatures,
registered modules, and tensor computations are retained for compatibility.
"""

from typing import Dict
import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, getCSM
    from .csms6s import CrossScan, CrossMerge
    from .csms6s import (
        CrossScan_Ab_1direction,
        CrossMerge_Ab_1direction,
        CrossScan_Ab_2direction,
        CrossMerge_Ab_2direction,
    )
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, getCSM
    from csms6s import CrossScan, CrossMerge
    from csms6s import (
        CrossScan_Ab_1direction,
        CrossMerge_Ab_1direction,
        CrossScan_Ab_2direction,
        CrossMerge_Ab_2direction,
    )
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(
            self.weight.shape
        )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class mamba_init:
    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=0.0001,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** (-0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class SS2Dviewscan:
    """Visual selective scan and its projection/normalization layers."""

    def __initv2__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=0.0001,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardv2

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[: -len(tag)]
            return (ret, value)

        self.disable_force32, forward_type = checkpostfix(
            "_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix(
            "_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix(
            "_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix(
            "_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix(
            "_onsigmoid", forward_type)
        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_cnorm:
            self.out_norm = nn.Sequential(
                LayerNorm(d_inner),
                nn.Identity() if channel_first else Permute(0, 3, 1, 2),
                nn.Conv2d(
                    d_inner,
                    d_inner,
                    kernel_size=3,
                    padding=1,
                    groups=d_inner,
                    bias=False,
                ),
                nn.Identity() if channel_first else Permute(0, 2, 3, 1),
            )
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                nn.Identity() if channel_first else Permute(0, 3, 1, 2),
                nn.Conv2d(
                    d_inner,
                    d_inner,
                    kernel_size=3,
                    padding=1,
                    groups=d_inner,
                    bias=False,
                ),
                nn.Identity() if channel_first else Permute(0, 2, 3, 1),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=-1 if channel_first else 1)
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = LayerNorm(d_inner)
        FORWARD_TYPES = dict(
            v01=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanMamba,
            ),
            v02=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanMamba,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v03=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v04=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v05=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v051d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                CrossScan=getCSM(1)[0],
                CrossMerge=getCSM(1)[1],
            ),
            v052d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                CrossScan=getCSM(2)[0],
                CrossMerge=getCSM(2)[1],
            ),
            v052dc=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                cascade2d=True,
            ),
            v2=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanCore,
            ),
            v3=partial(
                self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex
            ),
            v31d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScan_Ab_1direction,
                CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScan_Ab_2direction,
                CrossMerge=CrossMerge_Ab_2direction,
            ),
            v32dc=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                cascade2d=True,
            ),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 1
        d_proj = d_inner if self.disable_z else d_inner * 2
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs
            )
        self.x_proj = [
            nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if initialize in ["v0"]:
            self.dt_projs = [
                self.dt_init(
                    dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor
                )
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0)
            )
            self.dt_projs_bias = nn.Parameter(
                torch.stack([t.bias for t in self.dt_projs], dim=0)
            )
            del self.dt_projs
            self.A_logs = self.A_log_init(
                d_state, d_inner, copies=k_group, merge=True)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((k_group, d_inner, dt_rank))
            )
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.rand((k_group, d_inner, dt_rank))
            )
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        to_dtype=True,
        force_fp32=False,
        ssoflex=True,
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,
        cascade2d=False,
        **kwargs
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(
            u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True
        ):
            return SelectiveScan.apply(
                u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex
            )

        if cascade2d:

            def scan_rowcol(
                x: torch.Tensor,
                proj_weight: torch.Tensor,
                proj_bias: torch.Tensor,
                dt_weight: torch.Tensor,
                dt_bias: torch.Tensor,
                _As: torch.Tensor,
                _Ds: torch.Tensor,
                width=True,
            ):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = (XB * XH, XD, XW)
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = (XB * XW, XD, XH)
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)
                if no_einsum:
                    x_dbl = F.conv1d(
                        xs.view(_B, -1, _L),
                        proj_weight.view(-1, _D, 1),
                        bias=proj_bias.view(-1) if proj_bias is not None else None,
                        groups=2,
                    )
                    dts, Bs, Cs = torch.split(
                        x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2
                    )
                    dts = F.conv1d(
                        dts.contiguous().view(_B, -1, _L),
                        dt_weight.view(2 * _D, -1, 1),
                        groups=2,
                    )
                else:
                    x_dbl = torch.einsum(
                        "b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum(
                        "b k r l, k d r -> b k d l", dts, dt_weight)
                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)
                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)
                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            x = (
                F.layer_norm(x.permute(0, 2, 3, 1),
                             normalized_shape=(int(x.shape[1]),))
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            y_row = (
                scan_rowcol(
                    x,
                    proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                    proj_bias=(
                        x_proj_bias.view(4, -1)[:2].contiguous()
                        if x_proj_bias is not None
                        else None
                    ),
                    dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                    dt_bias=(
                        dt_projs_bias.view(4, -1)[:2].contiguous()
                        if dt_projs_bias is not None
                        else None
                    ),
                    _As=As[:2].contiguous().view(-1, N),
                    _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                    width=True,
                )
                .view(B, H, 2, -1, W)
                .sum(dim=2)
                .permute(0, 2, 1, 3)
            )
            y_row = (
                F.layer_norm(
                    y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            y_col = (
                scan_rowcol(
                    y_row,
                    proj_weight=x_proj_weight.view(4, -1, D)[2:]
                    .contiguous()
                    .to(y_row.dtype),
                    proj_bias=(
                        x_proj_bias.view(
                            4, -1)[2:].contiguous().to(y_row.dtype)
                        if x_proj_bias is not None
                        else None
                    ),
                    dt_weight=dt_projs_weight.view(4, D, -1)[2:]
                    .contiguous()
                    .to(y_row.dtype),
                    dt_bias=(
                        dt_projs_bias.view(
                            4, -1)[2:].contiguous().to(y_row.dtype)
                        if dt_projs_bias is not None
                        else None
                    ),
                    _As=As[2:].contiguous().view(-1, N),
                    _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                    width=False,
                )
                .view(B, W, 2, -1, H)
                .sum(dim=2)
                .permute(0, 2, 3, 1)
            )
            y = y_col
        else:
            xs = x
            if no_einsum:
                x_dbl = F.conv1d(
                    xs.view(B, -1, L),
                    x_proj_weight.view(-1, D, 1),
                    bias=x_proj_bias.view(-1) if x_proj_bias is not None else None,
                    groups=K,
                )
                dts, Bs, Cs = torch.split(
                    x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(
                    dts.contiguous().view(B, -1, L),
                    dt_projs_weight.view(K * D, -1, 1),
                    groups=K,
                )
            else:
                x_dbl = torch.einsum(
                    "b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum(
                    "b k r l, k d r -> b k d l", dts, dt_projs_weight)
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            y = ys
            if getattr(self, "__DEBUG__", False):
                setattr(
                    self,
                    "__data__",
                    dict(
                        A_logs=A_logs,
                        Bs=Bs,
                        Cs=Cs,
                        Ds=Ds,
                        us=xs,
                        dts=dts,
                        delta_bias=delta_bias,
                        ys=ys,
                        y=y,
                        H=H,
                        W=W,
                    ),
                )
        y = y.view(B, -1, H, W)
        if not channel_first:
            y = (
                y.view(B, -1, H * W)
                .transpose(dim0=1, dim1=2)
                .contiguous()
                .view(B, H, W, -1)
            )
        y = out_norm(y)
        return y.to(x.dtype) if to_dtype else y

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1 if self.channel_first else -1)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2Dview(nn.Module, mamba_init, SS2Dviewscan):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=0.0001,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class VSSBlockview(nn.Module):
    """Visual Mamba block accepting tokens of shape (B, H * W, C)."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        is_light_sr: bool = False,
        **kwargs
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2Dview(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        B, _, C = input.shape
        input = input.view(B, *x_size, C).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x.view(B, -1, C).contiguous()
        return x


class Mamba_Decoderview(nn.Module):
    def __init__(self, out_channels=3, dim=128, num_blocks=[4, 4], bias=False):
        super(Mamba_Decoderview, self).__init__()
        self.reduce_channel = nn.Conv2d(
            int(dim), int(dim / 2), kernel_size=1, bias=bias
        )
        drop_path_rate = 0.0
        norm_layer = "LN"
        channel_first = norm_layer.lower() in ["bn", "ln2d"]
        ssm_ratio = 1.0
        ssm_d_state = 8
        ssm_dt_rank = "auto"
        ssm_act_layer = "silu"
        ssm_conv = 3
        ssm_conv_bias = True
        ssm_drop_rate = 0.0
        ssm_init = "v0"
        forward_type = "v05"
        _NORMLAYERS = dict(
            ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU,
                          relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        self.encoder_level2 = nn.ModuleList(
            [
                VSSBlockview(
                    hidden_dim=dim,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.encoder_level1 = nn.ModuleList(
            [
                VSSBlockview(
                    hidden_dim=dim,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input_common):
        _, _, H, W = input_common.shape
        x = torch.cat([input1, input2], dim=1)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        out_enc_level2 = x
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H, W])
        out_enc_level2 = rearrange(
            out_enc_level2, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        out_enc_level2 = self.reduce_channel(out_enc_level2)
        x_123 = torch.cat([out_enc_level2, input_common], dim=1)
        x_123 = rearrange(x_123, "b c h w -> b (h w) c").contiguous()
        out_enc_level1 = x_123
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, (H, W))
        out_enc_level1 = rearrange(
            out_enc_level1, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1)


class SS2Dviewscan_cross:
    """Cross scan; the v05 path derives delta, B, and C from text features."""

    def __initv2__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=0.0001,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardv2

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[: -len(tag)]
            return (ret, value)

        self.disable_force32, forward_type = checkpostfix(
            "_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix(
            "_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix(
            "_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix(
            "_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix(
            "_onsigmoid", forward_type)
        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_cnorm:
            self.out_norm = nn.Sequential(
                LayerNorm(d_inner),
                nn.Identity() if channel_first else Permute(0, 3, 1, 2),
                nn.Conv2d(
                    d_inner,
                    d_inner,
                    kernel_size=3,
                    padding=1,
                    groups=d_inner,
                    bias=False,
                ),
                nn.Identity() if channel_first else Permute(0, 2, 3, 1),
            )
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                nn.Identity() if channel_first else Permute(0, 3, 1, 2),
                nn.Conv2d(
                    d_inner,
                    d_inner,
                    kernel_size=3,
                    padding=1,
                    groups=d_inner,
                    bias=False,
                ),
                nn.Identity() if channel_first else Permute(0, 2, 3, 1),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=-1 if channel_first else 1)
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = LayerNorm(d_inner)
        FORWARD_TYPES = dict(
            v01=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanMamba,
            ),
            v02=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanMamba,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v03=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v04=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v05=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                CrossScan=CrossScanTriton,
                CrossMerge=CrossMergeTriton,
            ),
            v051d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                CrossScan=getCSM(1)[0],
                CrossMerge=getCSM(1)[1],
            ),
            v052d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                CrossScan=getCSM(2)[0],
                CrossMerge=getCSM(2)[1],
            ),
            v052dc=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                no_einsum=True,
                cascade2d=True,
            ),
            v2=partial(
                self.forward_corev2,
                force_fp32=not self.disable_force32,
                SelectiveScan=SelectiveScanCore,
            ),
            v3=partial(
                self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex
            ),
            v31d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScan_Ab_1direction,
                CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                CrossScan=CrossScan_Ab_2direction,
                CrossMerge=CrossMerge_Ab_2direction,
            ),
            v32dc=partial(
                self.forward_corev2,
                force_fp32=False,
                SelectiveScan=SelectiveScanOflex,
                cascade2d=True,
            ),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 1
        d_proj = d_inner if self.disable_z else d_inner * 2
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.in_proj2 = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs
            )
            self.conv2d2 = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs
            )
        self.x_proj = [
            nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if initialize in ["v0"]:
            self.dt_projs = [
                self.dt_init(
                    dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor
                )
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0)
            )
            self.dt_projs_bias = nn.Parameter(
                torch.stack([t.bias for t in self.dt_projs], dim=0)
            )
            del self.dt_projs
            self.A_logs = self.A_log_init(
                d_state, d_inner, copies=k_group, merge=True)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((k_group, d_inner, dt_rank))
            )
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.rand((k_group, d_inner, dt_rank))
            )
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        xtext: torch.Tensor = None,
        to_dtype=True,
        force_fp32=False,
        ssoflex=True,
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,
        cascade2d=False,
        **kwargs
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(
            u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True
        ):
            return SelectiveScan.apply(
                u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex
            )

        if cascade2d:

            def scan_rowcol(
                x: torch.Tensor,
                proj_weight: torch.Tensor,
                proj_bias: torch.Tensor,
                dt_weight: torch.Tensor,
                dt_bias: torch.Tensor,
                _As: torch.Tensor,
                _Ds: torch.Tensor,
                width=True,
            ):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = (XB * XH, XD, XW)
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = (XB * XW, XD, XH)
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)
                if no_einsum:
                    x_dbl = F.conv1d(
                        xs.view(_B, -1, _L),
                        proj_weight.view(-1, _D, 1),
                        bias=proj_bias.view(-1) if proj_bias is not None else None,
                        groups=2,
                    )
                    dts, Bs, Cs = torch.split(
                        x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2
                    )
                    dts = F.conv1d(
                        dts.contiguous().view(_B, -1, _L),
                        dt_weight.view(2 * _D, -1, 1),
                        groups=2,
                    )
                else:
                    x_dbl = torch.einsum(
                        "b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum(
                        "b k r l, k d r -> b k d l", dts, dt_weight)
                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)
                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)
                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            x = (
                F.layer_norm(x.permute(0, 2, 3, 1),
                             normalized_shape=(int(x.shape[1]),))
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            y_row = (
                scan_rowcol(
                    x,
                    proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                    proj_bias=(
                        x_proj_bias.view(4, -1)[:2].contiguous()
                        if x_proj_bias is not None
                        else None
                    ),
                    dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                    dt_bias=(
                        dt_projs_bias.view(4, -1)[:2].contiguous()
                        if dt_projs_bias is not None
                        else None
                    ),
                    _As=As[:2].contiguous().view(-1, N),
                    _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                    width=True,
                )
                .view(B, H, 2, -1, W)
                .sum(dim=2)
                .permute(0, 2, 1, 3)
            )
            y_row = (
                F.layer_norm(
                    y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            y_col = (
                scan_rowcol(
                    y_row,
                    proj_weight=x_proj_weight.view(4, -1, D)[2:]
                    .contiguous()
                    .to(y_row.dtype),
                    proj_bias=(
                        x_proj_bias.view(
                            4, -1)[2:].contiguous().to(y_row.dtype)
                        if x_proj_bias is not None
                        else None
                    ),
                    dt_weight=dt_projs_weight.view(4, D, -1)[2:]
                    .contiguous()
                    .to(y_row.dtype),
                    dt_bias=(
                        dt_projs_bias.view(
                            4, -1)[2:].contiguous().to(y_row.dtype)
                        if dt_projs_bias is not None
                        else None
                    ),
                    _As=As[2:].contiguous().view(-1, N),
                    _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                    width=False,
                )
                .view(B, W, 2, -1, H)
                .sum(dim=2)
                .permute(0, 2, 3, 1)
            )
            y = y_col
        else:

            xs = xtext
            print("xs", xs.shape)
            if no_einsum:
                x_dbl = F.conv1d(
                    xs.view(B, -1, L),
                    x_proj_weight.view(-1, D, 1),
                    bias=x_proj_bias.view(-1) if x_proj_bias is not None else None,
                    groups=K,
                )
                dts, Bs, Cs = torch.split(
                    x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(
                    dts.contiguous().view(B, -1, L),
                    dt_projs_weight.view(K * D, -1, 1),
                    groups=K,
                )
            else:
                x_dbl = torch.einsum(
                    "b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum(
                    "b k r l, k d r -> b k d l", dts, dt_projs_weight)
            xs = x
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            y = ys
            if getattr(self, "__DEBUG__", False):
                setattr(
                    self,
                    "__data__",
                    dict(
                        A_logs=A_logs,
                        Bs=Bs,
                        Cs=Cs,
                        Ds=Ds,
                        us=xs,
                        dts=dts,
                        delta_bias=delta_bias,
                        ys=ys,
                        y=y,
                        H=H,
                        W=W,
                    ),
                )
        y = y.view(B, -1, H, W)
        if not channel_first:
            y = (
                y.view(B, -1, H * W)
                .transpose(dim0=1, dim1=2)
                .contiguous()
                .view(B, H, W, -1)
            )
        y = out_norm(y)
        return y.to(x.dtype) if to_dtype else y

    def forwardv2(self, x: torch.Tensor, xtext: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        xtext = self.in_proj2(xtext)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            xtext = xtext.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
            xtext = self.conv2d2(xtext)
        x = self.act(x)
        xtext = self.act(xtext)
        y = self.forward_core(x, xtext)
        y = self.out_act(y)
        out = self.dropout(self.out_proj(y))
        return out


class SS2Dcross(nn.Module, mamba_init, SS2Dviewscan_cross):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=0.0001,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class VSSBlockcross(nn.Module):
    """Text-conditioned visual block with global channel modulation.

    Both inputs have shape (B, H * W, C). The scan output is spatially
    averaged, L1-normalized, and used to modulate projected visual features.
    """

    def __init__(
        self,
        input_channel=64,
        mambadim=64,
        bias=False,
        drop_path_rate=0,
        dim=64,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=1.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        is_light_sr: bool = False,
        **kwargs
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channel, mambadim, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(mambadim, mambadim, kernel_size=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(2 * mambadim, mambadim, kernel_size=1)
        self.prelu3 = nn.PReLU()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.self_attention = VSSBlockview(
            hidden_dim=dim,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type="v05",
        )
        self.cross_attention = SS2Dcross(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.imagefeature2textfeature = nn.Conv2d(
            mambadim, hidden_dim, kernel_size=1)
        self.drop_path = DropPath(drop_path)

    def forward(self, input, text, x_size):
        B, _, C = input.shape
        [H, W] = x_size
        input = rearrange(input, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        input = self.prelu1(self.conv1(input))
        input = rearrange(input, "b c h w -> b (h w) c", h=H, w=W).contiguous()
        input = input.view(B, *x_size, C).contiguous()
        text = text.view(B, *x_size, C).contiguous()
        input = self.ln_1(input)
        xtext = self.ln_2(text)
        input = input.view(B, -1, C).contiguous()
        input = self.self_attention(input, x_size)
        input = input.view(B, *x_size, C).contiguous()
        input_sideout = input
        input = input.view(B, -1, C).contiguous()
        input = rearrange(input, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        input2text = self.imagefeature2textfeature(input)
        input2text = rearrange(
            input2text, "b c h w -> b (h w) c", h=H, w=W
        ).contiguous()
        input2text = input2text.view(B, *x_size, C).contiguous()
        input = rearrange(input, "b c h w -> b (h w) c", h=H, w=W).contiguous()
        input = input.view(B, *x_size, C).contiguous()
        ca_out = self.drop_path(self.cross_attention(input2text, xtext))
        ca_out = ca_out.view(B, -1, C).contiguous()

        ca_out = torch.nn.functional.adaptive_avg_pool1d(
            ca_out.permute(0, 2, 1), 1
        ).permute(0, 2, 1)
        ca_out = F.normalize(ca_out, p=1, dim=2)
        input2text = input2text.view(B, -1, C).contiguous()
        ca_out = input2text * ca_out
        input2text = input2text.view(B, *x_size, C).contiguous()
        ca_out = rearrange(ca_out, "b (h w) c -> b c h w",
                           h=H, w=W).contiguous()
        input_sideout = input_sideout.view(B, -1, C).contiguous()
        input_sideout = rearrange(
            input_sideout, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        x_out = self.prelu3(
            self.conv3(
                torch.cat(
                    (input_sideout, self.prelu2(self.conv2(ca_out)) + input_sideout),
                    dim=1,
                )
            )
        )
        x_out = rearrange(x_out, "b c h w -> b (h w) c", h=H, w=W).contiguous()
        x_out = x_out.view(B, -1, C).contiguous()
        return x_out


class FeatureWiseAffine(nn.Module):
    """Legacy text projection and sequence-length matching module.

    The historical name is retained for compatibility. This module performs
    a 1x1 Conv1d projection followed by F.pad, rather than affine modulation.
    Input text has shape (B, T, in_channels); output is (B, L, out_channels),
    where L = x.shape[1]. Positive padding adds zeros; negative padding crops.
    """

    def __init__(self, in_channels, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, text_embed):
        text_embedding = self.conv(text_embed.permute(0, 2, 1))
        transposed_tensor = text_embedding.permute(0, 2, 1)
        transposed_tensor = F.pad(
            transposed_tensor,
            (0, 0, 0, x.shape[1] - transposed_tensor.shape[1], 0, 0),
            "constant",
            0,
        )
        return transposed_tensor


class Mamba_Text_Enhance(nn.Module):
    """Enhance two private branches and one common branch with text.

    Visual inputs: (B, H * W, C). Text inputs: (B, T, 768).
    Returns the three enhanced visual token sequences in the same order.
    The original configuration uses C = 64.
    """

    def __init__(self, dim=64, bias=False):
        super(Mamba_Text_Enhance, self).__init__()
        drop_path_rate = 0.0
        norm_layer = "LN"
        channel_first = norm_layer.lower() in ["bn", "ln2d"]
        ssm_d_state = 8
        ssm_dt_rank = "auto"
        ssm_act_layer = "silu"
        ssm_conv = 3
        ssm_conv_bias = True
        ssm_drop_rate = 0.0
        ssm_init = "v0"
        forward_type_text = "v05_noz"
        ssm_ratio_text = 2.0
        _NORMLAYERS = dict(
            ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU,
                          relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        self.text_preprocess = FeatureWiseAffine(
            in_channels=768, out_channels=dim)
        self.textcross1_0 = VSSBlockcross(
            hidden_dim=dim,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio_text,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type_text,
        )
        self.textcross2_0 = VSSBlockcross(
            hidden_dim=dim,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio_text,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type_text,
        )
        self.textcrosscommon_0 = VSSBlockcross(
            hidden_dim=dim,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio_text,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type_text,
        )

    def forward(
        self,
        image1_unique,
        image2_unique,
        image_common,
        text1_unique,
        text2_unique,
        text_common,
        H,
        W,
    ):
        input1_text_match = self.text_preprocess(image1_unique, text1_unique)
        input2_text_match = self.text_preprocess(image2_unique, text2_unique)
        input_common_text_match = self.text_preprocess(
            image_common, text_common)
        image1_unique = self.textcross1_0(
            image1_unique, input1_text_match, [H, W])
        image2_unique = self.textcross2_0(
            image2_unique, input2_text_match, [H, W])
        input_common = self.textcrosscommon_0(
            image_common, input_common_text_match, [H, W]
        )
        return (image1_unique, image2_unique, input_common)


class Mamba_Decoder_textguide(nn.Module):
    """Text-guided fusion decoder for two private maps and a common map.

    The original configuration uses three (B, 64, H, W) visual maps and
    three (B, T, 768) text sequences. Output is (B, out_channels, H, W).
    """

    def __init__(self, out_channels=3, dim=64, num_blocks=[4, 4], bias=False):
        super(Mamba_Decoder_textguide, self).__init__()
        drop_path_rate = 0.0
        norm_layer = "LN"
        channel_first = norm_layer.lower() in ["bn", "ln2d"]
        ssm_ratio = 1.0
        ssm_d_state = 8
        ssm_dt_rank = "auto"
        ssm_act_layer = "silu"
        ssm_conv = 3
        ssm_conv_bias = True
        ssm_drop_rate = 0.0
        ssm_init = "v0"
        forward_type = "v05"
        _NORMLAYERS = dict(
            ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU,
                          relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        self.reduce_channel = nn.Conv2d(
            int(dim * 2), int(dim), kernel_size=1, bias=bias
        )
        self.text_enhance = Mamba_Text_Enhance()
        self.encoder_level2 = nn.ModuleList(
            [
                VSSBlockview(
                    hidden_dim=dim * 2,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.encoder_level1 = nn.ModuleList(
            [
                VSSBlockview(
                    hidden_dim=dim * 2,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim * 2), int(dim), kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input_common, text1, text2, text_common):
        _, _, H, W = input_common.shape
        input1 = rearrange(input1, "b c h w -> b (h w) c").contiguous()
        input2 = rearrange(input2, "b c h w -> b (h w) c").contiguous()
        input_common = rearrange(
            input_common, "b c h w -> b (h w) c").contiguous()
        input1, input2, input_common = self.text_enhance(
            input1, input2, input_common, text1, text2, text_common, H, W
        )
        input1 = rearrange(input1, "b (h w) c -> b c h w",
                           h=H, w=W).contiguous()
        input2 = rearrange(input2, "b (h w) c -> b c h w",
                           h=H, w=W).contiguous()
        input_common = rearrange(
            input_common, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        fusionFeature = torch.cat([input1, input2], dim=1)
        fusionFeature = rearrange(
            fusionFeature, "b c h w -> b (h w) c").contiguous()
        out_enc_level2 = fusionFeature
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H, W])
        out_enc_level2 = rearrange(
            out_enc_level2, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        out_enc_level2 = self.reduce_channel(out_enc_level2)
        fusionFeature2 = torch.cat([out_enc_level2, input_common], dim=1)
        fusionFeature2 = rearrange(
            fusionFeature2, "b c h w -> b (h w) c").contiguous()
        out_enc_level1 = fusionFeature2
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, (H, W))
        out_enc_level1 = rearrange(
            out_enc_level1, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1)


class Mamba_Encoder(nn.Module):
    def __init__(
        self, inp_channels=6, out_channels=3, dim=128, num_blocks=[4, 4], bias=False
    ):
        super(Mamba_Encoder, self).__init__()
        drop_path_rate = 0.0
        norm_layer = "LN"
        channel_first = norm_layer.lower() in ["bn", "ln2d"]
        ssm_ratio = 1.0
        ssm_d_state = 8
        ssm_dt_rank = "auto"
        ssm_act_layer = "silu"
        ssm_conv = 3
        ssm_conv_bias = True
        ssm_drop_rate = 0.0
        ssm_init = "v0"
        forward_type = "v05"
        _NORMLAYERS = dict(
            ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU,
                          relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        self.encoder_level2 = nn.ModuleList(
            [
                VSSBlockview(
                    hidden_dim=64,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.proj = nn.Conv2d(
            inp_channels, 64, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, input):
        _, _, H, W = input.shape
        x = self.proj(input)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        out_enc_level2 = x
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H, W])
        out_enc_level2 = rearrange(
            out_enc_level2, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        return out_enc_level2


class Mamba_Decoder(nn.Module):
    def __init__(self, dim=128, out_channels=3, bias=False):
        super(Mamba_Decoder, self).__init__()
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.output(input)
        return self.sigmoid(out)


class Mamba_Decoder2(nn.Module):
    def __init__(self, dim=128, out_channels=3, num_blocks=[4, 4], bias=False):
        super(Mamba_Decoder2, self).__init__()
        drop_path_rate = 0.0
        norm_layer = "LN"
        channel_first = norm_layer.lower() in ["bn", "ln2d"]
        ssm_ratio = 1.0
        ssm_d_state = 8
        ssm_dt_rank = "auto"
        ssm_act_layer = "silu"
        ssm_conv = 3
        ssm_conv_bias = True
        ssm_drop_rate = 0.0
        ssm_init = "v0"
        forward_type = "v05"
        _NORMLAYERS = dict(
            ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU,
                          relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        self.encoder_level2 = nn.ModuleList(
            [
                VSSBlockview(
                    hidden_dim=128,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        _, _, H, W = input.shape
        x = input
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        out_enc_level2 = x
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H, W])
        out_enc_level2 = rearrange(
            out_enc_level2, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        out = self.output(out_enc_level2)
        return self.sigmoid(out)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 dim: int = 128,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64,
                 out_channels: int = 1,):
        super(UNet, self).__init__()
        self.in_channels = dim
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(dim, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return self.sigmoid(x)
