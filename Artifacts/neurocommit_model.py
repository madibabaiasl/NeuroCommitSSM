# neurocommit_model.py
# ------------------------------------------------------------
# Inference on NEW Phase-4 caches (*.preproc.v4.3.npz)
# Loads bundle:
#   model_dir/final_model.pt
#   model_dir/final_cfg.json
#   model_dir/final_thresholds.json
# And stats_fold.json for normalization
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


# -------------------------
# Safe SDPA math-only helper (matches your training style)
# -------------------------
def sdp_math_only():
    if not torch.cuda.is_available():
        return nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        pass
    try:
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
    except Exception:
        pass
    return nullcontext()


# -------------------------
# Scenarios (same meaning as your training)
# S0=all, S1=no EEG, S2=no EMG, S3=no ET, S4=EEG only, S5=EMG only, S6=ET only
# -------------------------
SCENARIO_KEEP = {
    "S0": (1, 1, 1),
    "S1": (0, 1, 1),
    "S2": (1, 0, 1),
    "S3": (1, 1, 0),
    "S4": (1, 0, 0),
    "S5": (0, 1, 0),
    "S6": (0, 0, 1),
}


def apply_scenario_inplace(batch: Dict[str, torch.Tensor], scenario: str):
    if scenario not in SCENARIO_KEEP:
        raise ValueError(f"Unknown scenario: {scenario} (expected one of {list(SCENARIO_KEEP.keys())})")
    ke, km, kt = SCENARIO_KEEP[scenario]

    if ke == 0:
        batch["X_EEG"].zero_(); batch["M_EEG"].zero_()
        batch["r_eeg"].zero_()
    if km == 0:
        batch["X_EMG"].zero_(); batch["M_EMG"].zero_()
        batch["r_emg"].zero_()
    if kt == 0:
        batch["X_ET"].zero_(); batch["M_ET"].zero_()
        batch["r_et"].zero_()
    return batch


# ============================================================
# Config (same as training)
# ============================================================
@dataclass
class NeuroCommitCfg:
    d_model: int = 160
    drop: float = 0.10
    patch: int = 25
    rel_dim: int = 4

    eeg_virtual_K: int = 6
    eeg_sinc_filters: int = 6
    eeg_graph_heads: int = 4

    emg_synergy_M: int = 4
    emg_env_kernel: int = 25
    emg_burst_kernel: int = 9

    et_event_bins: int = 4
    et_event_topk: int = 3

    fuse_use_uncertainty: bool = True
    fuse_ssm_hidden: int = 192
    fuse_ssm_layers: int = 1

    commit_thr_on: float = 0.80
    commit_thr_off: float = 0.55
    commit_dwell_windows: int = 2
    commit_cooldown_windows: int = 0

    commit_hidden: int = 160
    commit_attn_pool: bool = True

    proj_dim: int = 160

    # Phase 5.5 features (kept in model to match checkpoint)
    use_p55_features: bool = True
    feat_dim_psd: int = 96
    feat_dim_emg: int = 24
    feat_dim_mask: int = 3
    feat_z_scale: float = 0.25
    feat_token_scale: float = 0.25


# ============================================================
# Utilities
# ============================================================
def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def patchify_time_mask(M: torch.Tensor, patch: int, thr: float = 0.10) -> torch.Tensor:
    B, T, C = M.shape
    T2 = (T // patch) * patch
    M = M[:, :T2, :]
    m_time = (M.mean(dim=-1) > 0).float()
    L = T2 // patch
    m_patch = m_time.reshape(B, L, patch).mean(dim=-1)
    return (m_patch > thr).float()


# ============================================================
# Patch projection (same as training)
# ============================================================
class PatchProject1D(nn.Module):
    def __init__(self, Cin: int, D: int, patch: int, drop: float):
        super().__init__()
        self.patch = int(patch)
        self.proj = nn.Conv1d(Cin, D, kernel_size=self.patch, stride=self.patch, bias=False)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(drop)

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = X.shape
        p = self.patch
        T2 = (T // p) * p
        if T2 <= 0:
            raise RuntimeError("T too small for patch size")
        X = X[:, :T2, :]
        M = M[:, :T2, :]

        x = (X * M).transpose(1, 2)
        h = self.proj(self.drop(x)).transpose(1, 2)     # (B,L,D)
        h = self.norm(h)

        m_patch = patchify_time_mask(M, patch=p, thr=0.10)  # (B,L)
        return h, m_patch


# ============================================================
# EEG encoder (same as training)
# ============================================================
class DilatedDWGLUBlock(nn.Module):
    def __init__(self, C: int, k: int = 9, d: int = 1, drop: float = 0.1):
        super().__init__()
        pad = (k // 2) * d
        self.dw = nn.Conv1d(C, C, kernel_size=k, padding=pad, dilation=d, groups=C, bias=False)
        self.pw = nn.Conv1d(C, 2 * C, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, C), num_channels=C)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        a, b = y.chunk(2, dim=1)
        y = F.gelu(a) * torch.sigmoid(b)
        y = self.drop(y)
        return self.norm(x + y)


class TokenGraphMixer(nn.Module):
    def __init__(self, K: int, drop: float = 0.1):
        super().__init__()
        self.q = nn.Linear(K, K, bias=False)
        self.k = nn.Linear(K, K, bias=False)
        self.proj = nn.Linear(K, K, bias=False)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(K)

    def forward(self, Xv: torch.Tensor, Mv: torch.Tensor) -> torch.Tensor:
        mp = (Mv > 0).float()
        den = mp.sum(dim=1).clamp_min(1.0)
        s = (Xv * mp).sum(dim=1) / den

        q = self.q(s)
        k = self.k(s)
        A = torch.softmax((q.unsqueeze(-1) @ k.unsqueeze(-2)) / math.sqrt(Xv.shape[-1]), dim=-1)

        Y = torch.einsum("bti,bij->btj", Xv, A)
        Y = self.drop(self.proj(Y))
        return self.norm(Xv + Y)


def _masked_channel_standardize(X: torch.Tensor, M: torch.Tensor, eps: float = 1e-5):
    m = (M > 0).float()
    den = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu = (X * m).sum(dim=1, keepdim=True) / den
    var = ((X - mu) ** 2 * m).sum(dim=1, keepdim=True) / den
    Xz = (X - mu) / torch.sqrt(var + eps)
    return Xz * m


class EEG_VEMGraphBank(nn.Module):
    def __init__(self, Ce: int, cfg: NeuroCommitCfg):
        super().__init__()
        assert Ce == 8, "EEG encoder expects 8ch EEG"

        self.Ce = int(Ce)
        self.K  = int(cfg.eeg_virtual_K)
        self.Fb = int(cfg.eeg_sinc_filters)
        self.D  = int(cfg.d_model)
        self.drop = float(cfg.drop)

        ks_all = [9, 17, 33, 65, 129, 161]
        ks = ks_all[:max(1, min(self.Fb, len(ks_all)))]
        self.ms = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.Ce, self.Ce, kernel_size=k, padding=k//2, groups=self.Ce, bias=False),
                nn.GELU()
            )
            for k in ks
        ])
        self.ms_mix = nn.Conv1d(self.Ce * len(ks), self.Ce, kernel_size=1, bias=False)
        self.ms_norm = nn.GroupNorm(num_groups=self.Ce, num_channels=self.Ce)

        self.low = nn.Conv1d(self.Ce, self.Ce, kernel_size=151, padding=75, groups=self.Ce, bias=False)

        desc_dim = 6
        self.token_mlp = nn.Sequential(
            nn.Linear(desc_dim, 64),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(64, self.K),
        )

        self.token_graph = TokenGraphMixer(self.K, drop=self.drop)

        n_layers = int(max(1, getattr(cfg, "eeg_graph_heads", 3)))
        dilations = [1, 2, 4, 8, 1, 2][:n_layers]
        self.temporal = nn.Sequential(*[
            DilatedDWGLUBlock(C=self.K, k=9, d=int(d), drop=self.drop)
            for d in dilations
        ])

        self.patch = PatchProject1D(Cin=self.K, D=self.D, patch=cfg.patch, drop=self.drop)

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, Ce = X.shape
        assert Ce == self.Ce

        Xz = _masked_channel_standardize(X, M)

        x = Xz.transpose(1, 2)
        feats = [f(x) for f in self.ms]
        fb = torch.cat(feats, dim=1)
        xf = self.ms_norm(self.ms_mix(fb))
        xf_t = xf.transpose(1, 2)

        avail = M.mean(dim=1).clamp(0, 1)

        m = (M > 0).float()
        den = m.sum(dim=1).clamp_min(1.0)
        mu = (xf_t * m).sum(dim=1) / den
        var = (((xf_t - mu.unsqueeze(1)) ** 2) * m).sum(dim=1) / den
        std = torch.sqrt(var + 1e-6)
        energy = ((xf_t ** 2) * m).sum(dim=1) / den

        low = self.low(xf).transpose(1, 2)
        d1 = (low[:, 1:, :] - low[:, :-1, :]).abs()
        absdiff = (d1 * m[:, 1:, :]).sum(dim=1) / (m[:, 1:, :].sum(dim=1).clamp_min(1.0))
        slope = absdiff

        desc = torch.stack([mu, std, energy, slope, absdiff, avail], dim=-1)

        logits = self.token_mlp(desc)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits + torch.log(avail.unsqueeze(1).clamp_min(1e-6))
        W = torch.softmax(logits, dim=-1)

        Xv = torch.einsum("btc,bkc->btk", xf_t, W)
        Mv = torch.einsum("btc,bkc->btk", M, W).clamp(0, 1)

        Xv = self.token_graph(Xv, Mv)
        Xv_c = Xv.transpose(1, 2)
        Xv_c = self.temporal(Xv_c)
        Xv = Xv_c.transpose(1, 2)

        H, m_patch = self.patch(Xv, Mv)
        return H, m_patch


# ============================================================
# EMG encoder (same as training)
# ============================================================
class EMG_SyEMB(nn.Module):
    def __init__(self, Cm: int, cfg: NeuroCommitCfg):
        super().__init__()
        assert Cm == 4, "EMG_SyEMB expects 4ch EMG"
        self.Msyn = int(cfg.emg_synergy_M)
        self.D = int(cfg.d_model)
        self.drop = float(cfg.drop)

        k_env = int(cfg.emg_env_kernel)
        k_bur = int(cfg.emg_burst_kernel)

        self.smooth_env = nn.Conv1d(Cm, Cm, kernel_size=k_env, padding=k_env // 2, groups=Cm, bias=False)
        self.smooth_burst = nn.Conv1d(Cm, Cm, kernel_size=k_bur, padding=k_bur // 2, groups=Cm, bias=False)

        self.synergy_raw = nn.Parameter(torch.zeros(self.Msyn, Cm))

        self.pre = nn.Sequential(
            nn.Conv1d(self.Msyn, self.Msyn, kernel_size=9, padding=4, groups=self.Msyn, bias=False),
            nn.GELU(),
            nn.Conv1d(self.Msyn, self.Msyn, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(self.drop),
        )

        self.patch = PatchProject1D(Cin=self.Msyn, D=self.D, patch=cfg.patch, drop=self.drop)

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (X * M).transpose(1, 2)

        env = self.smooth_env(torch.abs(x))
        dx = torch.zeros_like(x)
        dx[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        burst = self.smooth_burst(torch.abs(dx))

        xm = 0.9 * env + 0.6 * burst + 0.2 * x

        W = F.softplus(self.synergy_raw)
        W = W / (W.sum(dim=1, keepdim=True).clamp_min(1e-6))
        S = torch.einsum("mc,bct->bmt", W, xm)

        Mm = (M.transpose(1, 2) > 0).float()
        Sm = torch.einsum("mc,bct->bmt", W, Mm).clamp(0, 1)

        S = self.pre(S).transpose(1, 2)
        Sm = Sm.transpose(1, 2)

        H, m_patch = self.patch(S, Sm)
        return H, m_patch


# ============================================================
# ET encoder (same as training)  <-- IMPORTANT for your load error
# ============================================================
class MaskedPatchify1D(nn.Module):
    def __init__(self, D: int, patch: int):
        super().__init__()
        self.patch = int(patch)
        self.conv = nn.Conv1d(D, D, kernel_size=self.patch, stride=self.patch, bias=False)
        self.ln = nn.LayerNorm(D)

    def forward(self, Hd: torch.Tensor, m_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = Hd.shape
        p = self.patch
        T2 = (T // p) * p
        Hd = Hd[:, :T2, :]
        m_time = m_time[:, :T2]

        x = Hd.transpose(1, 2)
        Hp = self.conv(x).transpose(1, 2)

        mv = F.avg_pool1d(m_time.unsqueeze(1), kernel_size=p, stride=p).squeeze(1)
        m_patch = (mv > 0.10).float()

        Hp = Hp / mv.clamp_min(0.10).unsqueeze(-1)
        Hp = self.ln(Hp)
        return Hp, m_patch


class VBTransformerBlock(nn.Module):
    def __init__(self, D: int, nhead: int = 4, drop: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=nhead, dropout=drop, batch_first=True)
        self.ln1 = nn.LayerNorm(D)
        self.ff = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4 * D, D),
            nn.Dropout(drop),
        )
        self.ln2 = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kpm = key_padding_mask
        if kpm is not None:
            kpm = kpm.to(torch.bool)
            if kpm.ndim == 2:
                all_mask = kpm.all(dim=1)
                if all_mask.any():
                    kpm = kpm.clone()
                    kpm[all_mask, 0] = False

        if x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                y, _ = self.attn(x.float(), x.float(), x.float(), key_padding_mask=kpm, need_weights=False)
            y = y.to(x.dtype)
        else:
            y, _ = self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)

        x = self.ln1(x + y)
        x = self.ln2(x + self.ff(x))
        return x


class ET_ECPT(nn.Module):
    def __init__(self, Ct: int, cfg: NeuroCommitCfg):
        super().__init__()
        self.Ct = int(Ct)
        self.D = int(cfg.d_model)
        self.drop = float(cfg.drop)
        self.patch = int(cfg.patch)

        self.topk = int(cfg.et_event_topk)
        self.event_bins = int(cfg.et_event_bins)
        self.nhead = 4

        self.dense = nn.Sequential(
            nn.Linear(self.Ct + 3, self.D),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Dropout(self.drop),
        )

        self.patchify = MaskedPatchify1D(D=self.D, patch=self.patch)

        self.ev_emb = nn.Embedding(self.event_bins * 2, self.D)
        self.ev_mag = nn.Sequential(nn.Linear(1, self.D), nn.Tanh())
        self.ev_ln = nn.LayerNorm(self.D)

        self.cross = nn.MultiheadAttention(embed_dim=self.D, num_heads=self.nhead, dropout=self.drop, batch_first=True)
        self.cross_ln = nn.LayerNorm(self.D)

        self.self1 = VBTransformerBlock(D=self.D, nhead=self.nhead, drop=self.drop)

        self.gate = nn.Sequential(
            nn.Linear(self.D * 2, self.D),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.D, self.D),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = X.shape
        p = self.patch
        T2 = (T // p) * p
        if T2 <= 0:
            return X.new_zeros((B, 0, self.D)), X.new_zeros((B, 0))

        X = X[:, :T2, :]
        M = M[:, :T2, :]

        v = M.mean(dim=-1, keepdim=True).clamp(0, 1)
        m_time = (v.squeeze(-1) > 0.05).float()

        dx = torch.zeros_like(X)
        dx[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
        dd = torch.zeros_like(X)
        dd[:, 2:, :] = dx[:, 2:, :] - dx[:, 1:-1, :]

        vel = torch.sqrt((dx * dx).mean(dim=-1, keepdim=True).clamp_min(1e-12))
        acc = torch.sqrt((dd * dd).mean(dim=-1, keepdim=True).clamp_min(1e-12))

        Xd = torch.cat([X, vel, acc, v], dim=-1)
        Hd = self.dense(Xd) * v

        Hp, m_patch = self.patchify(Hd, m_time)
        patch_kpm = (m_patch <= 0.0)

        e = (0.7 * vel + 0.3 * acc) * v
        e_flat = e.squeeze(-1)
        e_flat = e_flat.masked_fill(m_time <= 0.0, -1e9)

        k = min(self.topk, T2)
        if k <= 0:
            H = self.self1(Hp, key_padding_mask=patch_kpm)
            H = H * m_patch.unsqueeze(-1)
            return H, m_patch

        vals, idx = torch.topk(e_flat, k=k, dim=1, largest=True)
        vals = vals.clamp_min(0.0)

        bin_w = max(1, T2 // self.event_bins)
        bin_id = torch.clamp(idx // bin_w, min=0, max=self.event_bins - 1)

        dx_mean = dx.mean(dim=-1)
        sign = (dx_mean.gather(1, idx) >= 0).long()
        ev_id = (bin_id * 2 + sign).long()

        ev = self.ev_emb(ev_id) + self.ev_mag(vals.unsqueeze(-1))
        ev = self.ev_ln(ev)
        ev = ev * (vals > 0).float().unsqueeze(-1)

        if Hp.is_cuda:
            with sdp_math_only():
                with torch.autocast(device_type="cuda", enabled=False):
                    inj, _ = self.cross(query=Hp.float(), key=ev.float(), value=ev.float(), need_weights=False)
            inj = inj.to(Hp.dtype)
        else:
            inj, _ = self.cross(query=Hp, key=ev, value=ev, need_weights=False)

        Hp2 = self.cross_ln(Hp + inj)

        g = self.gate(torch.cat([Hp2, inj], dim=-1))
        H = Hp2 + g * inj

        H = self.self1(H, key_padding_mask=patch_kpm)
        H = H * m_patch.unsqueeze(-1)
        return H, m_patch


# ============================================================
# Fusion + commit head (same as training)
# ============================================================
class UncertaintyFusionSSM(nn.Module):
    def __init__(self, cfg: NeuroCommitCfg):
        super().__init__()
        D = int(cfg.d_model)
        self.rel_dim = int(cfg.rel_dim)
        self.use_unc = bool(cfg.fuse_use_uncertainty)

        self.u_mlp = nn.Sequential(
            nn.Linear(self.rel_dim + 1, D),
            nn.GELU(),
            nn.Dropout(float(cfg.drop)),
            nn.Linear(D, 1),
        )

        self.gru = nn.GRU(
            input_size=D,
            hidden_size=int(cfg.fuse_ssm_hidden),
            num_layers=int(cfg.fuse_ssm_layers),
            batch_first=True,
            bidirectional=False
        )
        self.out = nn.Sequential(
            nn.LayerNorm(int(cfg.fuse_ssm_hidden)),
            nn.Linear(int(cfg.fuse_ssm_hidden), D),
            nn.GELU(),
            nn.Dropout(float(cfg.drop)),
            nn.Linear(D, D),
        )

    def forward(
        self,
        He: torch.Tensor, Hm: torch.Tensor, Ht: torch.Tensor,
        me: torch.Tensor, mm: torch.Tensor, mt: torch.Tensor,
        r_eeg: torch.Tensor, r_emg: torch.Tensor, r_et: torch.Tensor,
    ):
        B, L, D = He.shape

        ae = me.mean(dim=1, keepdim=True)
        am = mm.mean(dim=1, keepdim=True)
        at = mt.mean(dim=1, keepdim=True)

        if self.use_unc:
            ue = self.u_mlp(torch.cat([r_eeg, ae], dim=1))
            um = self.u_mlp(torch.cat([r_emg, am], dim=1))
            ut = self.u_mlp(torch.cat([r_et,  at], dim=1))
            U = torch.cat([ue, um, ut], dim=1)
            A = torch.cat([ae, am, at], dim=1).clamp(0, 1)
            A = (A > 0.05).float()
            logits = -U + torch.log(A.clamp_min(1e-6))
            w = torch.softmax(logits, dim=1)
        else:
            A = torch.cat([(ae > 0.05).float(), (am > 0.05).float(), (at > 0.05).float()], dim=1)
            w = A / (A.sum(dim=1, keepdim=True).clamp_min(1e-12))

        w_e = w[:, 0].view(B, 1, 1)
        w_m = w[:, 1].view(B, 1, 1)
        w_t = w[:, 2].view(B, 1, 1)

        Hf = w_e * He + w_m * Hm + w_t * Ht

        out_seq, hN = self.gru(Hf)
        h = hN[-1]
        z = self.out(h)

        st = {
            "w_eeg": w[:, 0],
            "w_emg": w[:, 1],
            "w_et":  w[:, 2],
            "avail_eeg": ae.squeeze(1),
            "avail_emg": am.squeeze(1),
            "avail_et":  at.squeeze(1),
        }
        return z, Hf, st


class CommitDecisionHeadTemporal(nn.Module):
    def __init__(self, cfg: NeuroCommitCfg):
        super().__init__()
        D = int(cfg.d_model)
        H = int(cfg.commit_hidden)
        self.drop = float(cfg.drop)
        self.use_attn_pool = bool(cfg.commit_attn_pool)

        self.gru = nn.GRU(D, H, num_layers=1, batch_first=True, bidirectional=False)

        if self.use_attn_pool:
            self.attn = nn.Sequential(
                nn.Linear(H, H),
                nn.GELU(),
                nn.Dropout(self.drop),
                nn.Linear(H, 1),
            )

        self.mlp = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(H, 1),
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(H, 2),
        )

    def forward(self, Hseq: torch.Tensor, m_seq: Optional[torch.Tensor] = None):
        out, h_last = self.gru(Hseq)   # out: (B,T,H)

        if self.use_attn_pool:
            a = self.attn(out).squeeze(-1)     # (B,T)
            a_fp32 = a.float()

            if m_seq is not None:
                mask = (m_seq <= 0)
                all_masked = mask.all(dim=1)
                if all_masked.any():
                    mask = mask.clone()
                    mask[all_masked, 0] = False
                a_fp32 = a_fp32.masked_fill(mask, torch.finfo(a_fp32.dtype).min)

            w = torch.softmax(a_fp32, dim=1).to(out.dtype).unsqueeze(-1)  # (B,T,1)
            hT = (out * w).sum(dim=1)  # (B,H)
        else:
            hT = h_last[-1]

        ct_logit = self.mlp(hT).squeeze(-1)  # (B,)
        ct = torch.sigmoid(ct_logit)
        logits_state = self.cls(hT)

        stab = out.var(dim=1, unbiased=False).mean(dim=-1)
        d = (out[:, 1:, :] - out[:, :-1, :]).abs().mean(dim=(1, 2)) if out.shape[1] > 1 else out.new_zeros((out.shape[0],))
        aux = {"stability_var": stab, "delta_mean": d}
        return ct, ct_logit, logits_state, aux


class NeuroCommitM3(nn.Module):
    def __init__(self, Ce: int, Cm: int, Ct: int, cfg: NeuroCommitCfg, num_task: int = 5):
        super().__init__()
        self.cfg = cfg
        self.num_task = int(num_task)

        self.eeg = EEG_VEMGraphBank(Ce, cfg)
        self.emg = EMG_SyEMB(Cm, cfg)
        self.et  = ET_ECPT(Ct, cfg)

        self.fuse = UncertaintyFusionSSM(cfg)

        D = int(cfg.d_model)
        self.head_action = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(float(cfg.drop)),
            nn.Linear(D, 2),
        )
        self.head_task = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(float(cfg.drop)),
            nn.Linear(D, self.num_task),
        )

        self.commit = CommitDecisionHeadTemporal(cfg)

        # SSL heads (kept to match checkpoint keys)
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, int(cfg.proj_dim)),
            nn.GELU(),
            nn.Linear(int(cfg.proj_dim), int(cfg.proj_dim)),
        )
        self.dec_eeg = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.dec_emg = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.dec_et  = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

        feat_dim = int(cfg.feat_dim_psd + cfg.feat_dim_emg + cfg.feat_dim_mask)
        self.feat_dim = feat_dim
        self.use_feats = bool(cfg.use_p55_features)

        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, D),
            nn.GELU(),
            nn.Dropout(float(cfg.drop)),
            nn.Linear(D, D),
        )

    def forward_window(self, batch: Dict[str, torch.Tensor]):
        Xe, Xm, Xt = batch["X_EEG"], batch["X_EMG"], batch["X_ET"]
        Me, Mm, Mt = batch["M_EEG"], batch["M_EMG"], batch["M_ET"]
        r_eeg = batch["r_eeg"].float()
        r_emg = batch["r_emg"].float()
        r_et  = batch["r_et"].float()

        # features are optional; if not present, skip feature injection
        if self.use_feats and ("F_psd" in batch) and ("F_emg" in batch) and ("F_mask" in batch):
            F_cat = torch.cat([batch["F_psd"], batch["F_emg"], batch["F_mask"]], dim=1).float()
        else:
            F_cat = Xe.new_zeros((Xe.shape[0], 0))

        He, me = self.eeg(Xe, Me)
        Hm, mm = self.emg(Xm, Mm)
        Ht, mt = self.et (Xt, Mt)

        z, Hf, st = self.fuse(He, Hm, Ht, me, mm, mt, r_eeg, r_emg, r_et)

        # feature injection only if F_cat has correct dim
        if self.use_feats and (F_cat is not None) and (F_cat.numel() > 0) and (F_cat.shape[1] == self.feat_dim):
            femb = self.feat_proj(F_cat)  # (B,D)
            z = z + float(self.cfg.feat_z_scale) * femb
            ftok = float(self.cfg.feat_token_scale) * femb.unsqueeze(1)  # (B,1,D)
            Hf = torch.cat([Hf, ftok], dim=1)
            me = torch.cat([me, torch.ones((me.shape[0], 1), device=me.device, dtype=me.dtype)], dim=1)
            mm = torch.cat([mm, torch.ones((mm.shape[0], 1), device=mm.device, dtype=mm.dtype)], dim=1)
            mt = torch.cat([mt, torch.ones((mt.shape[0], 1), device=mt.device, dtype=mt.dtype)], dim=1)

        logits_action = self.head_action(z)
        logits_task   = self.head_task(z)

        m_fuse = ((me + mm + mt) > 0).float().clamp(0, 1)
        ct, ct_logit, logits_state, aux = self.commit(Hf, m_seq=m_fuse)

        st2 = dict(st)
        st2.update({
            "commit_logits": logits_state,
            "ct_logit": ct_logit,
            "stability_var": aux["stability_var"],
            "delta_mean": aux["delta_mean"],
        })
        return logits_action, logits_task, ct, st2


# ============================================================
# Commit Decision Filter (optional stream state)
# ============================================================
class CommitDecisionFilter:
    def __init__(self, thr_on: float = 0.80, thr_off: float = 0.55, dwell_windows: int = 2, cooldown_windows: int = 0):
        self.thr_on = float(thr_on)
        self.thr_off = float(thr_off)
        self.dwell = int(max(1, dwell_windows))
        self.cooldown = int(max(0, cooldown_windows))
        self.state = "HOLD"
        self._on_count = 0
        self._cooldown_left = 0

    def reset(self):
        self.state = "HOLD"
        self._on_count = 0
        self._cooldown_left = 0

    def step(self, ct: float) -> Tuple[str, bool]:
        ct = float(ct)
        commit_event = False

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if self.state == "HOLD":
            if self._cooldown_left > 0:
                self._on_count = 0
                return self.state, False

            if ct >= self.thr_on:
                self._on_count += 1
            else:
                self._on_count = 0

            if self._on_count >= self.dwell:
                self.state = "COMMIT"
                commit_event = True
                self._on_count = 0
            return self.state, commit_event

        if ct <= self.thr_off:
            self.state = "HOLD"
            if self.cooldown > 0:
                self._cooldown_left = self.cooldown
        return self.state, False


# ============================================================
# Phase-4 cache -> window tensors
# ============================================================
def _as_np(x) -> np.ndarray:
    return np.asarray(x)

def _pad_or_trim_cols(X: np.ndarray, target_C: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Expected 2D array [T,C]")
    T, C = X.shape
    if C == target_C:
        return X
    if C > target_C:
        return X[:, :target_C]
    out = np.zeros((T, target_C), dtype=X.dtype)
    out[:, :C] = X
    return out

def load_phase4_cache(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    # expected keys from your Phase-4:
    # EEG, EEG_mask, EMG_env, EMG_mask, ET, ET_mask, fs, t
    return out

def _find_stats_arrays(stats: Dict[str, Any], name: str, C: int, aliases: Tuple[str, ...] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tries to find mean/std for a modality in many common stats_fold.json layouts:
      1) stats["stats"][name]["mean"/"std"]
      2) stats[name]["mean"/"std"]
      3) flat arrays like stats[f"{name}_mean"], stats[f"{name}_std"]
    """
    names = (name,) + tuple(aliases)

    def _pad_trim(v: np.ndarray, fill: float) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        out = np.full((C,), fill, dtype=np.float32)
        out[:min(C, v.size)] = v[:min(C, v.size)]
        return out

    # ------------- try nested dict blocks -------------
    containers = []
    if isinstance(stats, dict):
        containers.append(("root", stats))
        if isinstance(stats.get("stats", None), dict):
            containers.append(("root.stats", stats["stats"]))
        if isinstance(stats.get("norm", None), dict):
            containers.append(("root.norm", stats["norm"]))
        if isinstance(stats.get("normalization", None), dict):
            containers.append(("root.normalization", stats["normalization"]))

    for cname, cont in containers:
        for nm in names:
            blk = cont.get(nm, None) if isinstance(cont, dict) else None
            if isinstance(blk, dict):
                mean = blk.get("mean", blk.get("mu", None))
                std  = blk.get("std",  blk.get("sigma", None))
                if mean is not None and std is not None:
                    mu = _pad_trim(mean, 0.0)
                    sd = _pad_trim(std, 1.0)
                    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
                    print(f"[norm] using {cname}['{nm}']['mean/std']")
                    return mu, sd

    # ------------- try flat arrays -------------
    for nm in names:
        mean_key = f"{nm}_mean"
        std_key  = f"{nm}_std"
        if mean_key in stats and std_key in stats:
            mu = _pad_trim(stats[mean_key], 0.0)
            sd = _pad_trim(stats[std_key], 1.0)
            sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
            print(f"[norm] using root['{mean_key}/{std_key}']")
            return mu, sd

    print(f"[warn] Could not find {name} mean/std in stats_fold.json. Using mean=0 std=1 for {name}.")
    return np.zeros((C,), dtype=np.float32), np.ones((C,), dtype=np.float32)


def normalize_window(X: np.ndarray, M: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32).reshape(1, 1, -1)
    sd = np.asarray(sd, dtype=np.float32).reshape(1, 1, -1)
    Xn = (X - mu) / sd
    return Xn * M

def quality_proxy_vec(X: np.ndarray, M: np.ndarray, rel_dim: int = 4) -> np.ndarray:
    """
    Simple reliability proxy (compatible shape with training):
    [valid, missing, flatline, clip] padded/truncated to rel_dim.
    """
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)

    valid = float(M.mean())
    missing = float(1.0 - valid)

    # flatline proxy: channel std extremely small on valid samples
    flat_thr = 1e-6
    flat_flags = []
    for c in range(X.shape[-1]):
        mc = M[..., c].reshape(-1) > 0.5
        if mc.sum() < 10:
            flat_flags.append(1.0)
            continue
        xc = X[..., c].reshape(-1)[mc]
        flat_flags.append(1.0 if float(np.std(xc)) < flat_thr else 0.0)
    flatline = float(np.mean(flat_flags)) if flat_flags else 0.0

    # clip proxy: robust outlier fraction (very conservative)
    clip_flags = []
    for c in range(X.shape[-1]):
        mc = M[..., c].reshape(-1) > 0.5
        if mc.sum() < 10:
            clip_flags.append(0.0)
            continue
        xc = X[..., c].reshape(-1)[mc]
        med = np.median(xc)
        mad = np.median(np.abs(xc - med)) + 1e-9
        z = (xc - med) / (1.4826 * mad)
        clip_flags.append(float(np.mean(np.abs(z) > 10.0)))
    clip = float(np.mean(clip_flags)) if clip_flags else 0.0

    r = np.array([valid, missing, flatline, clip], dtype=np.float32)
    if rel_dim <= 4:
        return r[:rel_dim]
    out = np.zeros((rel_dim,), dtype=np.float32)
    out[:4] = r
    return out

def iter_windows_from_cache(
    cache: Dict[str, Any],
    win_len: int = 500,
    stride: int = 62,
    Ce: int = 8,
    Cm: int = 4,
    Ct: int = 15,
):
    fs = float(cache.get("fs", 250.0))
    t = np.asarray(cache.get("t", None), dtype=np.float64)
    EEG = np.asarray(cache.get("EEG"), dtype=np.float32)
    EEGm = np.asarray(cache.get("EEG_mask"), dtype=np.float32)

    EMG = np.asarray(cache.get("EMG_env"), dtype=np.float32)
    EMGm = np.asarray(cache.get("EMG_mask"), dtype=np.float32)

    ET = np.asarray(cache.get("ET"), dtype=np.float32)
    ETm = np.asarray(cache.get("ET_mask"), dtype=np.float32)

    # pad/trim channels if needed
    EEG = _pad_or_trim_cols(EEG, Ce); EEGm = _pad_or_trim_cols(EEGm, Ce)
    EMG = _pad_or_trim_cols(EMG, Cm); EMGm = _pad_or_trim_cols(EMGm, Cm)
    ET  = _pad_or_trim_cols(ET, Ct);  ETm  = _pad_or_trim_cols(ETm, Ct)

    N = int(EEG.shape[0])
    if N < win_len:
        return

    y_action = cache.get("y_action", None)
    y_task = cache.get("y_task", None)
    if y_action is not None:
        y_action = np.asarray(y_action).reshape(-1)
    if y_task is not None:
        y_task = np.asarray(y_task).reshape(-1)

    win_idx = 0
    for s in range(0, N - win_len + 1, stride):
        e = s + win_len
        mid = s + win_len // 2
        center_time_s = float(t[mid]) if (t is not None and t.size > mid) else float(mid / fs)

        Xa = EEG[s:e, :][None, ...]   # (1,T,C)
        Xm_ = EEGm[s:e, :][None, ...]
        Xb = EMG[s:e, :][None, ...]
        Mm_ = EMGm[s:e, :][None, ...]
        Xc = ET[s:e, :][None, ...]
        Mt_ = ETm[s:e, :][None, ...]

        # GT (optional)
        gt_action = None
        gt_task = None
        if y_action is not None and y_action.size >= e:
            gt_action = int(np.mean(y_action[s:e]) >= 0.5)
        if y_task is not None and y_task.size >= e:
            # majority task among nonzero (raw task ids 0..5)
            seg = y_task[s:e].astype(np.int64)
            seg = seg[seg > 0]
            gt_task = int(np.bincount(seg).argmax()) if seg.size else 0

        yield {
            "win_idx": int(win_idx),
            "start": int(s),
            "center_time_s": float(center_time_s),
            "EEG": Xa, "EEG_mask": Xm_,
            "EMG": Xb, "EMG_mask": Mm_,
            "ET":  Xc, "ET_mask":  Mt_,
            "gt_action": gt_action,
            "gt_task": gt_task,
        }
        win_idx += 1


# ============================================================
# Bundle loading
# ============================================================
def torch_load_flex(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def load_bundle(model_dir: Path, stats_path: Path, device: str):
    model_dir = Path(model_dir)
    cfg_path = model_dir / "final_cfg.json"
    thr_path = model_dir / "final_thresholds.json"
    pt_path  = model_dir / "final_model.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing: {thr_path}")
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing: {pt_path}")
    if not Path(stats_path).exists():
        raise FileNotFoundError(f"Missing stats: {stats_path}")

    cfg_obj = json.loads(cfg_path.read_text())

    # Your final_cfg.json commonly stores cfgM + Ce/Cm/Ct + rel_dim
    if "cfgM" in cfg_obj and isinstance(cfg_obj["cfgM"], dict):
        cfgM_dict = cfg_obj["cfgM"]
    else:
        # fallback: assume whole file is cfg dict
        cfgM_dict = cfg_obj

    Ce = int(cfg_obj.get("Ce", 8))
    Cm = int(cfg_obj.get("Cm", 4))
    Ct = int(cfg_obj.get("Ct", cfg_obj.get("Ct", 15)))
    rel_dim = int(cfg_obj.get("rel_dim", cfgM_dict.get("rel_dim", 4)))

    cfgM_dict = dict(cfgM_dict)
    cfgM_dict["rel_dim"] = rel_dim

    cfg = NeuroCommitCfg(**cfgM_dict)
    model = NeuroCommitM3(Ce=Ce, Cm=Cm, Ct=Ct, cfg=cfg, num_task=5).to(device)

    sd_obj = torch_load_flex(pt_path, device=device)
    if isinstance(sd_obj, dict) and ("state_dict" in sd_obj):
        sd = sd_obj["state_dict"]
    else:
        sd = sd_obj

    model.load_state_dict(sd, strict=True)
    model.eval()

    thresholds = json.loads(thr_path.read_text())
    stats = json.loads(Path(stats_path).read_text())

    return model, cfg, thresholds, stats, Ce, Cm, Ct, rel_dim


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def run_inference(
    model: NeuroCommitM3,
    cfg: NeuroCommitCfg,
    thresholds: Dict[str, Any],
    stats: Dict[str, Any],
    cache_paths: List[Path],
    scenario: str,
    out_csv: Path,
    summary_csv: Optional[Path] = None,   # ✅ ADD THIS
    device: str = "cpu",
    batch_size: int = 128,
    win_len: int = 500,
    stride: int = 62,
):

    TASK_NAMES = ["T1", "T2", "T3", "T4", "T5"]

    Ce = 8
    Cm = 4
    Ct = None

    # infer Ct from model.et.Ct
    try:
        Ct = int(model.et.Ct)
    except Exception:
        Ct = 15

    # load normalization stats (robust key search)
    eeg_mu, eeg_sd = _find_stats_arrays(stats, "EEG", Ce, aliases=("eeg",))
    emg_mu, emg_sd = _find_stats_arrays(stats, "EMG_env", Cm, aliases=("EMG", "emg"))
    et_mu,  et_sd  = _find_stats_arrays(stats, "ET", Ct, aliases=("et",))


    # thresholds
    thr_action = float(thresholds.get("thr_action_by_scenario", {}).get(scenario, 0.5))
    thr_gate   = float(thresholds.get("thr_commit_gate_by_scenario", {}).get(scenario, thr_action))

    # commit filter params
    c_on  = float(thresholds.get("commit_thr_on", cfg.commit_thr_on))
    c_off = float(thresholds.get("commit_thr_off", cfg.commit_thr_off))
    c_dw  = int(thresholds.get("commit_dwell_windows", cfg.commit_dwell_windows))
    c_cd  = int(thresholds.get("commit_cooldown_windows", cfg.commit_cooldown_windows))

    print(f"[thr] scenario={scenario} thr_action={thr_action:.4f} thr_gate={thr_gate:.4f}")
    print(f"[commit] thr_on={c_on:.2f} thr_off={c_off:.2f} dwell={c_dw} cooldown={c_cd}")

    rows = []

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    for fp in cache_paths:
        cache = load_phase4_cache(fp)
        base = fp.name.replace(".preproc.v4.3.npz", "")
        # reset per-file filter
        filt = CommitDecisionFilter(thr_on=c_on, thr_off=c_off, dwell_windows=c_dw, cooldown_windows=c_cd)

        # collect windows, run in mini-batches
        buf = []
        meta_buf = []

        for w in iter_windows_from_cache(cache, win_len=win_len, stride=stride, Ce=Ce, Cm=Cm, Ct=Ct):
            # pack into tensors later
            buf.append(w)
            meta_buf.append({
                "file": str(fp),
                "stem": base,
                "win_idx": w["win_idx"],
                "center_time_s": w["center_time_s"],
                "gt_action": w["gt_action"],
                "gt_task": w["gt_task"],
            })

            if len(buf) >= batch_size:
                _flush_batch(model, cfg, scenario, buf, meta_buf, rows,
                             device, dev_type, thr_action, thr_gate,
                             eeg_mu, eeg_sd, emg_mu, emg_sd, et_mu, et_sd,
                             filt, TASK_NAMES)
                buf, meta_buf = [], []

        if buf:
            _flush_batch(model, cfg, scenario, buf, meta_buf, rows,
                         device, dev_type, thr_action, thr_gate,
                         eeg_mu, eeg_sd, emg_mu, emg_sd, et_mu, et_sd,
                         filt, TASK_NAMES)

        # per-file summary print
        df_f = pd.DataFrame([r for r in rows if r["stem"] == base])
        if len(df_f):
            n = len(df_f)
            act = int((df_f["pred_action"] == 1).sum())
            # majority predicted task among predicted action windows
            df_act = df_f[df_f["pred_action"] == 1]
            if len(df_act):
                pred_task = int(df_act["pred_task"].value_counts().idxmax())
            else:
                pred_task = 0
            print(f"[file] {base}: windows={n} pred_action={act} majority_pred_task={pred_task}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[done] wrote:", out_csv)


    # -----------------------------
    # Optional per-file summary CSV
    # -----------------------------
    if summary_csv is not None:
        df_all = pd.DataFrame(rows)

        # numeric GT columns safely (they may be "" strings)
        df_all["gt_action_num"] = pd.to_numeric(df_all.get("gt_action", np.nan), errors="coerce")
        df_all["gt_task_num"]   = pd.to_numeric(df_all.get("gt_task", np.nan), errors="coerce")

        def _mode_or_default(s: pd.Series, default=0):
            s = s.dropna()
            if len(s) == 0:
                return default
            m = s.mode()
            return int(m.iloc[0]) if len(m) else default

        per_file = []
        for stem, g in df_all.groupby("stem"):
            n = len(g)
            pred_action_rate = float((g["pred_action"] == 1).mean())

            # GT action rate (if available)
            if g["gt_action_num"].notna().any():
                gt_action_rate = float((g["gt_action_num"] >= 0.5).mean())
            else:
                gt_action_rate = np.nan

            # Majority GT task among GT action windows
            if g["gt_task_num"].notna().any():
                gt_task_majority = _mode_or_default(g.loc[g["gt_task_num"] > 0, "gt_task_num"], default=0)
            else:
                gt_task_majority = np.nan

            # Majority predicted task among predicted action windows
            pred_task_majority = _mode_or_default(g.loc[g["pred_action"] == 1, "pred_task"], default=0)

            # (optional extra) commit events per file
            commit_events = int(g.get("commit_event", pd.Series([0]*n)).sum())

            per_file.append({
                "stem": stem,
                "n_windows": int(n),
                "gt_action_rate": gt_action_rate,
                "pred_action_rate": pred_action_rate,
                "gt_task_majority": gt_task_majority,
                "pred_task_majority": pred_task_majority,
                "commit_events": commit_events,
            })

        df_pf = pd.DataFrame(per_file).sort_values("stem")
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df_pf.to_csv(summary_csv, index=False)
        print("[done] wrote per-file summary:", summary_csv)


    # also print top predicted tasks overall
    df = pd.DataFrame(rows)
    df_act = df[df["pred_action"] == 1]
    if len(df_act):
        print("\n[overall] top predicted tasks (action windows):")
        print(df_act["pred_task_name"].value_counts().head(10).to_string())
    else:
        print("\n[overall] no ACTION windows predicted.")


def _flush_batch(
    model: NeuroCommitM3,
    cfg: NeuroCommitCfg,
    scenario: str,
    buf: List[Dict[str, Any]],
    meta_buf: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    device: str,
    dev_type: str,
    thr_action: float,
    thr_gate: float,
    eeg_mu: np.ndarray, eeg_sd: np.ndarray,
    emg_mu: np.ndarray, emg_sd: np.ndarray,
    et_mu: np.ndarray,  et_sd: np.ndarray,
    filt: CommitDecisionFilter,
    TASK_NAMES: List[str],
):

    # stack
    # -----------------------------
    X_EEG = np.concatenate([b["EEG"] for b in buf], axis=0).astype(np.float32)
    M_EEG = np.concatenate([b["EEG_mask"] for b in buf], axis=0).astype(np.float32)

    X_EMG = np.concatenate([b["EMG"] for b in buf], axis=0).astype(np.float32)
    M_EMG = np.concatenate([b["EMG_mask"] for b in buf], axis=0).astype(np.float32)

    X_ET  = np.concatenate([b["ET"] for b in buf], axis=0).astype(np.float32)
    M_ET  = np.concatenate([b["ET_mask"] for b in buf], axis=0).astype(np.float32)

    # ✅ keep raw copies BEFORE normalization (for Phase-5.5)
    X_EEG_raw = X_EEG.copy()
    X_EMG_raw = X_EMG.copy()
    X_ET_raw  = X_ET.copy()

    B = int(X_EEG.shape[0])
    rel_dim = int(cfg.rel_dim)


    # normalize
    X_EEG = normalize_window(X_EEG, M_EEG, eeg_mu, eeg_sd)
    X_EMG = normalize_window(X_EMG, M_EMG, emg_mu, emg_sd)
    X_ET  = normalize_window(X_ET,  M_ET,  et_mu,  et_sd)

    # reliability proxies (computed on raw normalized tensors is OK; we just need stable signal)
    r_eeg = np.stack([quality_proxy_vec(X_EEG[i:i+1], M_EEG[i:i+1], rel_dim=rel_dim) for i in range(B)], axis=0)
    r_emg = np.stack([quality_proxy_vec(X_EMG[i:i+1], M_EMG[i:i+1], rel_dim=rel_dim) for i in range(B)], axis=0)
    r_et  = np.stack([quality_proxy_vec(X_ET[i:i+1],  M_ET[i:i+1],  rel_dim=rel_dim) for i in range(B)], axis=0)

    # -----------------------------
    # ✅ Phase-5.5 features (ON THE FLY)
    # -----------------------------
    use_p55 = bool(getattr(cfg, "use_p55_features", False))
    if use_p55:
        F_psd = np.zeros((B, 96), dtype=np.float32)
        F_emg = np.zeros((B, 24), dtype=np.float32)
        F_mask = np.zeros((B, 3), dtype=np.float32)

        for i in range(B):
            # X_*_raw[i] is (T,C) — correct for p55_*_one_window()
            F_psd[i]  = p55_eeg_feats_one_window(X_EEG_raw[i], M_EEG[i], fs=P55_FS)
            F_emg[i]  = p55_emg_feats_one_window(X_EMG_raw[i], M_EMG[i])
            F_mask[i] = p55_mask_feats_from_masks(M_EEG[i], M_EMG[i], M_ET[i])


    else:
        F_psd = F_emg = F_mask = None


    batch = {
        "X_EEG": torch.from_numpy(X_EEG).to(device),
        "M_EEG": torch.from_numpy(M_EEG).to(device),
        "X_EMG": torch.from_numpy(X_EMG).to(device),
        "M_EMG": torch.from_numpy(M_EMG).to(device),
        "X_ET":  torch.from_numpy(X_ET).to(device),
        "M_ET":  torch.from_numpy(M_ET).to(device),
        "r_eeg": torch.from_numpy(r_eeg).to(device),
        "r_emg": torch.from_numpy(r_emg).to(device),
        "r_et":  torch.from_numpy(r_et).to(device),
    }
    # ✅ ADD THIS BLOCK RIGHT HERE (after batch dict)
    if use_p55 and (F_psd is not None):
        batch["F_psd"]  = torch.from_numpy(F_psd).to(device)
        batch["F_emg"]  = torch.from_numpy(F_emg).to(device)
        batch["F_mask"] = torch.from_numpy(F_mask).to(device)


    apply_scenario_inplace(batch, scenario)

    if use_p55 and ("F_mask" in batch):
        ke, km, kt = SCENARIO_KEEP[scenario]
        if ke == 0:
            batch["F_psd"].zero_()
            batch["F_mask"][:, 0] = 0.0
        if km == 0:
            batch["F_emg"].zero_()
            batch["F_mask"][:, 1] = 0.0
        if kt == 0:
            batch["F_mask"][:, 2] = 0.0


    with sdp_math_only():
        with torch.autocast(device_type=dev_type, enabled=bool(dev_type == "cuda")):
            logits_a, logits_t, ct, st = model.forward_window(batch)

    pa = torch.softmax(logits_a, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32, copy=False)
    pt = torch.softmax(logits_t, dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)  # (B,5)
    ct_raw = ct.detach().cpu().numpy().astype(np.float32, copy=False)

    pred_action = (pa >= float(thr_action)).astype(np.int64)
    pred_taskK = np.argmax(pt, axis=-1).astype(np.int64)  # 0..4
    pred_task = (pred_taskK + 1).astype(np.int64)         # 1..5

    # if action not predicted, set task=0
    pred_task = np.where(pred_action == 1, pred_task, 0)

    # gate ct using thr_gate (commit gating)
    ct_eff = ct_raw * (pa >= float(thr_gate)).astype(np.float32)

    # commit filter state per-window (sequence order = batch order)
    states = []
    events = []
    for i in range(B):
        st_i, ev_i = filt.step(float(np.clip(ct_eff[i], 0.0, 1.0)))
        states.append(st_i)
        events.append(1 if ev_i else 0)

    # weights/availability diagnostics
    w_e = st.get("w_eeg", torch.zeros((B,), device=logits_a.device)).detach().cpu().numpy()
    w_m = st.get("w_emg", torch.zeros((B,), device=logits_a.device)).detach().cpu().numpy()
    w_t = st.get("w_et",  torch.zeros((B,), device=logits_a.device)).detach().cpu().numpy()

    a_e = st.get("avail_eeg", torch.zeros((B,), device=logits_a.device)).detach().cpu().numpy()
    a_m = st.get("avail_emg", torch.zeros((B,), device=logits_a.device)).detach().cpu().numpy()
    a_t = st.get("avail_et",  torch.zeros((B,), device=logits_a.device)).detach().cpu().numpy()

    for i in range(B):
        m = meta_buf[i]
        pt_i = pt[i]
        rows.append({
            "file": m["file"],
            "stem": m["stem"],
            "win_idx": int(m["win_idx"]),
            "center_time_s": float(m["center_time_s"]),

            "action_prob": float(pa[i]),
            "pred_action": int(pred_action[i]),

            "pred_task": int(pred_task[i]),
            "pred_task_name": (TASK_NAMES[int(pred_taskK[i])] if int(pred_action[i]) == 1 else "REST/NA"),
            "task_prob_T1": float(pt_i[0]),
            "task_prob_T2": float(pt_i[1]),
            "task_prob_T3": float(pt_i[2]),
            "task_prob_T4": float(pt_i[3]),
            "task_prob_T5": float(pt_i[4]),

            "ct_raw": float(ct_raw[i]),
            "ct_gated": float(ct_eff[i]),
            "commit_state": str(states[i]),
            "commit_event": int(events[i]),

            "w_eeg": float(w_e[i]),
            "w_emg": float(w_m[i]),
            "w_et":  float(w_t[i]),
            "avail_eeg": float(a_e[i]),
            "avail_emg": float(a_m[i]),
            "avail_et":  float(a_t[i]),

            "gt_action": (int(m["gt_action"]) if m["gt_action"] is not None else ""),
            "gt_task":   (int(m["gt_task"]) if m["gt_task"] is not None else ""),
        })

# ============================================================
# Phase-5.5 feature extraction (ON-THE-FLY) for NEW Phase-4 caches
#   EEG: 96 = 12 feats/ch * 8ch
#   EMG: 24 = 6 feats/ch  * 4ch
#   MASK: 3 = [eeg_mask_mean, emg_mask_mean, et_mask_mean]
# No SciPy dependency (numpy FFT Welch)
# ============================================================

P55_FS = 250.0
P55_WELCH_NPERSEG = 256
P55_EPS = 1e-8
P55_MIN_VALID_FRAC = 0.40
P55_MIN_VALID_SAMPLES = 64

P55_EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu":    (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

def _p55_safe_log(x: np.ndarray, eps: float = P55_EPS) -> np.ndarray:
    return np.log(np.maximum(x, eps))

def _p55_validity_ok(m: np.ndarray, y: np.ndarray) -> Tuple[bool, float, int]:
    m = (np.asarray(m) > 0.5)
    y = np.asarray(y)
    valid = m & np.isfinite(y)
    cnt = int(valid.sum())
    frac = float(cnt) / float(max(1, y.shape[0]))
    ok = (cnt >= int(P55_MIN_VALID_SAMPLES)) and (frac >= float(P55_MIN_VALID_FRAC))
    return ok, frac, cnt

def _p55_interp_fill_1d_strict(y: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, float]:
    y = np.asarray(y, np.float32)
    m = np.asarray(m, np.float32)

    ok, vfrac, _ = _p55_validity_ok(m, y)
    if not ok:
        return np.zeros_like(y, dtype=np.float32), vfrac

    m_bool = (m > 0.5) & np.isfinite(y)
    valid_idx = np.where(m_bool)[0]
    y_f = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    bad = ~m_bool
    if bad.any():
        xi = valid_idx.astype(np.float32)
        yi = y_f[valid_idx].astype(np.float32)
        x = np.arange(y.shape[0], dtype=np.float32)
        y_f[bad] = np.interp(x[bad], xi, yi).astype(np.float32)

    y_f = np.nan_to_num(y_f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y_f, vfrac

def _p55_prepare_window_strict(X: np.ndarray, M: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_filled: (T,C) filled signal
      valid_chan: (C,) 0/1 channel validity (strict)
    """
    X = np.asarray(X, np.float32)
    T, C = X.shape
    if C == 0 or T == 0:
        return X, np.zeros((C,), dtype=np.float32)

    if M is None:
        M = np.ones_like(X, dtype=np.float32)
    else:
        M = np.asarray(M, np.float32)
        if M.shape != X.shape:
            raise RuntimeError(f"P55 mask mismatch: X{X.shape} vs M{M.shape}")

    X_out = np.zeros_like(X, dtype=np.float32)
    valid_chan = np.zeros((C,), dtype=np.float32)

    for c in range(C):
        yc, vfrac = _p55_interp_fill_1d_strict(X[:, c], M[:, c])
        is_valid = (vfrac >= P55_MIN_VALID_FRAC) and (np.nanmax(np.abs(yc)) > 1e-12)
        valid_chan[c] = 1.0 if is_valid else 0.0
        X_out[:, c] = yc
    return X_out, valid_chan

def _welch_psd_np(x: np.ndarray, fs: float, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: (T,C) float32
    returns f: (F,), Pxx: (F,C)
    """
    x = np.asarray(x, np.float32)
    T, C = x.shape
    nperseg = int(min(max(8, nperseg), T))
    noverlap = int(min(max(0, noverlap), nperseg - 1))
    step = nperseg - noverlap
    if step <= 0:
        step = max(1, nperseg // 2)

    win = np.hanning(nperseg).astype(np.float32)
    scale = fs * float((win * win).sum() + 1e-12)

    nfft = nperseg
    F = nfft // 2 + 1
    Pacc = np.zeros((F, C), dtype=np.float32)
    nseg = 0

    for s in range(0, T - nperseg + 1, step):
        seg = x[s:s + nperseg, :]
        seg = seg - seg.mean(axis=0, keepdims=True)
        segw = seg * win[:, None]
        Xf = np.fft.rfft(segw, n=nfft, axis=0)
        P = (np.abs(Xf) ** 2).astype(np.float32) / float(scale)
        # one-sided correction (approx)
        if F > 2:
            P[1:-1, :] *= 2.0
        Pacc += P
        nseg += 1

    if nseg == 0:
        # fallback: one segment
        seg = x - x.mean(axis=0, keepdims=True)
        segw = seg[:nperseg, :] * win[:, None]
        Xf = np.fft.rfft(segw, n=nfft, axis=0)
        P = (np.abs(Xf) ** 2).astype(np.float32) / float(scale)
        if F > 2:
            P[1:-1, :] *= 2.0
        Pacc = P
        nseg = 1

    Pxx = Pacc / float(max(1, nseg))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs).astype(np.float32)
    return f, Pxx

def p55_eeg_feats_one_window(x: np.ndarray, m: Optional[np.ndarray], fs: float = P55_FS) -> np.ndarray:
    """
    Output: (96,) for 8ch EEG => 12 feats/ch * 8ch
    Feats per ch:
      - log abs bandpower delta/theta/mu/beta (4)
      - rel bandpower delta/theta/mu/beta (4)
      - spectral entropy 1-30 (1)
      - log Hjorth activity/mobility/complexity (3)
    """
    x = np.asarray(x, np.float32)
    T, C = x.shape
    if T == 0 or C == 0:
        return np.zeros((12 * C,), dtype=np.float32)

    x_fill, valid_chan = _p55_prepare_window_strict(x, m)

    f, Pxx = _welch_psd_np(x_fill, fs=float(fs), nperseg=P55_WELCH_NPERSEG, noverlap=P55_WELCH_NPERSEG // 2)

    m_1_30 = (f >= 1.0) & (f <= 30.0)
    if not np.any(m_1_30):
        return np.zeros((12 * C,), dtype=np.float32)

    f_1_30 = f[m_1_30]
    P_1_30 = Pxx[m_1_30, :]
    total_power = np.trapezoid(P_1_30, f_1_30, axis=0).astype(np.float32) + P55_EPS

    abs_band_feats, rel_band_feats = [], []
    for (lo, hi) in P55_EEG_BANDS.values():
        mband = (f >= lo) & (f <= hi)
        if not np.any(mband):
            bp = np.zeros((C,), dtype=np.float32)
        else:
            bp = np.trapezoid(Pxx[mband, :], f[mband], axis=0).astype(np.float32)
        abs_band_feats.append(_p55_safe_log(bp))
        rel_band_feats.append((bp / total_power).astype(np.float32))

    abs_band = np.stack(abs_band_feats, axis=0)  # (4,C)
    rel_band = np.stack(rel_band_feats, axis=0)  # (4,C)

    # spectral entropy (1-30)
    Psum = P_1_30.sum(axis=0, keepdims=True).astype(np.float32) + P55_EPS
    Pnorm = (P_1_30 / Psum).astype(np.float32)
    spec_entropy = -(Pnorm * _p55_safe_log(Pnorm)).sum(axis=0).astype(np.float32)  # (C,)
    spec_entropy_row = spec_entropy[None, :]

    # Hjorth
    dx = np.diff(x_fill, axis=0)
    ddx = np.diff(dx, axis=0)
    var_x = (np.var(x_fill, axis=0).astype(np.float32) + P55_EPS)
    var_dx = (np.var(dx, axis=0).astype(np.float32) + P55_EPS)
    var_ddx = (np.var(ddx, axis=0).astype(np.float32) + P55_EPS)

    activity = var_x
    mobility = np.sqrt(var_dx / var_x).astype(np.float32)
    mobility_dx = np.sqrt(var_ddx / var_dx).astype(np.float32)
    complexity = (mobility_dx / (mobility + P55_EPS)).astype(np.float32)

    hj = np.stack([activity, mobility, complexity], axis=0).astype(np.float32)
    hj = _p55_safe_log(hj)

    feats = np.concatenate([abs_band, rel_band, spec_entropy_row, hj], axis=0)  # (12,C)
    feats *= valid_chan[None, :]
    return feats.reshape(-1).astype(np.float32)

def _p55_zero_crossings(x: np.ndarray, thr: float = 0.01) -> int:
    if x.size < 2:
        return 0
    x1, x2 = x[:-1], x[1:]
    crosses = (x1 * x2) < 0
    crosses = crosses & (np.abs(x2 - x1) >= thr)
    return int(crosses.sum())

def _p55_ssc(x: np.ndarray, thr: float = 0.01) -> int:
    if x.size < 3:
        return 0
    x1, x2, x3 = x[:-2], x[1:-1], x[2:]
    cond = ((x2 - x1) * (x2 - x3)) > 0
    cond = cond & ((np.abs(x2 - x1) >= thr) | (np.abs(x3 - x2) >= thr))
    return int(cond.sum())

def p55_emg_feats_one_window(x: np.ndarray, m: Optional[np.ndarray]) -> np.ndarray:
    """
    Output: (24,) for 4ch EMG => 6 feats/ch
    [rms, mav, log(1+WL), log(1+ZC), log(1+SSC), log(1+var)]
    """
    x = np.asarray(x, np.float32)
    T, C = x.shape
    if T == 0 or C == 0:
        return np.zeros((6 * C,), dtype=np.float32)

    x_fill, valid_chan = _p55_prepare_window_strict(x, m)

    feats = np.zeros((C, 6), dtype=np.float32)
    for c in range(C):
        if valid_chan[c] < 0.5:
            continue
        sig = x_fill[:, c].astype(np.float32, copy=False)
        sig0 = sig - float(np.mean(sig))
        rms = float(np.sqrt(np.mean(sig0 * sig0) + P55_EPS))
        mav = float(np.mean(np.abs(sig0)))
        wl = float(np.sum(np.abs(np.diff(sig0))))
        zc = float(_p55_zero_crossings(sig0, thr=0.01))
        ssc = float(_p55_ssc(sig0, thr=0.01))
        varv = float(np.var(sig0))
        feats[c] = np.asarray([
            rms,
            mav,
            np.log(1.0 + max(wl, 0.0)),
            np.log(1.0 + max(zc, 0.0)),
            np.log(1.0 + max(ssc, 0.0)),
            np.log(1.0 + max(varv, 0.0)),
        ], dtype=np.float32)
    return feats.reshape(-1).astype(np.float32)

def p55_mask_feats_from_masks(M_eeg: np.ndarray, M_emg: np.ndarray, M_et: np.ndarray) -> np.ndarray:
    def _mean(M):
        M = np.asarray(M, np.float32)
        return float(M.mean()) if M.size else 0.0
    return np.asarray([_mean(M_eeg), _mean(M_emg), _mean(M_et)], dtype=np.float32)

# ============================================================
# Cache discovery
# ============================================================
def find_cache_files(data_root: Path, suffix: str = ".preproc.v4.3.npz") -> List[Path]:
    data_root = Path(data_root)
    if data_root.is_file() and data_root.name.endswith(suffix):
        return [data_root]
    # common: either pass label folder or the whole New_data folder
    paths = sorted(data_root.rglob(f"*{suffix}"))
    return paths

# ============================================================
# ✅ FIXED: load_bundle (strict=False + key report)
# ============================================================
def load_bundle(model_dir: Path, stats_path: Path, device: str):
    model_dir = Path(model_dir)
    cfg_path = model_dir / "final_cfg.json"
    thr_path = model_dir / "final_thresholds.json"
    pt_path  = model_dir / "final_model.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing: {thr_path}")
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing: {pt_path}")
    if not Path(stats_path).exists():
        raise FileNotFoundError(f"Missing stats: {stats_path}")

    cfg_obj = json.loads(cfg_path.read_text())

    # final_cfg.json structure: usually {"cfgM": {...}, "Ce":8,"Cm":4,"Ct":15,"rel_dim":4,...}
    if "cfgM" in cfg_obj and isinstance(cfg_obj["cfgM"], dict):
        cfgM_dict = dict(cfg_obj["cfgM"])
    else:
        cfgM_dict = dict(cfg_obj)

    Ce = int(cfg_obj.get("Ce", 8))
    Cm = int(cfg_obj.get("Cm", 4))
    Ct = int(cfg_obj.get("Ct", cfgM_dict.get("Ct", 15)))
    rel_dim = int(cfg_obj.get("rel_dim", cfgM_dict.get("rel_dim", 4)))

    cfgM_dict["rel_dim"] = rel_dim
    cfg = NeuroCommitCfg(**cfgM_dict)

    model = NeuroCommitM3(Ce=Ce, Cm=Cm, Ct=Ct, cfg=cfg, num_task=5).to(device)

    # load checkpoint
    sd_obj = torch_load_flex(pt_path, device=device)
    if isinstance(sd_obj, dict) and ("state_dict" in sd_obj):
        sd = sd_obj["state_dict"]
    else:
        sd = sd_obj

    # ✅ IMPORTANT: strict=False to avoid crash; then report keys
    incomp = model.load_state_dict(sd, strict=False)
    missing = incomp.missing_keys
    unexpected = incomp.unexpected_keys


    print("[bundle] loaded:", pt_path)
    print("[bundle] strict=False")
    if missing:
        print(f"[bundle] missing_keys ({len(missing)}):", missing[:12], ("..." if len(missing) > 12 else ""))
    else:
        print("[bundle] missing_keys: 0")

    if unexpected:
        print(f"[bundle] unexpected_keys ({len(unexpected)}):", unexpected[:12], ("..." if len(unexpected) > 12 else ""))
    else:
        print("[bundle] unexpected_keys: 0")

    model.eval()

    thresholds = json.loads(thr_path.read_text())
    stats = json.loads(Path(stats_path).read_text())

    return model, cfg, thresholds, stats, Ce, Cm, Ct, rel_dim


# ============================================================
# CLI
# ============================================================
# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--scenario", type=str, default="S0")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--summary_csv", type=str, default="", help="Optional per-file summary CSV")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--win_len", type=int, default=500)
    ap.add_argument("--stride", type=int, default=62)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[warn] cuda requested but not available; using cpu.")
        device = "cpu"
    print("[device]", device)

    model, cfg, thresholds, stats, Ce, Cm, Ct, rel_dim = load_bundle(Path(args.model_dir), Path(args.stats), device=device)

    cache_paths = find_cache_files(Path(args.data_root), suffix=".preproc.v4.3.npz")
    if not cache_paths:
        raise FileNotFoundError(f"No caches found under {args.data_root} (expected *.preproc.v4.3.npz)")

    print(f"[data] found caches: {len(cache_paths)}")
    print(" first 5:", [p.name for p in cache_paths[:5]])

    summary_csv = Path(args.summary_csv) if args.summary_csv.strip() else None

    run_inference(
        model=model,
        cfg=cfg,
        thresholds=thresholds,
        stats=stats,
        cache_paths=cache_paths,
        scenario=args.scenario,
        out_csv=Path(args.out_csv),
        summary_csv=summary_csv,
        device=device,
        batch_size=int(args.batch_size),
        win_len=int(args.win_len),
        stride=int(args.stride),
    )

if __name__ == "__main__":
    main()

