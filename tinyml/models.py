
import math, os, json, random
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
@_register_model
class RegularCNN1D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, **kw):
        super().__init__()
        if 'RegularCNN' in globals():
            self.core = RegularCNN(in_ch, num_classes, **kw)
        else:
            # fallback to your shared core if RegularCNN isn’t present
            self.core = SharedCoreSeparable1D(in_ch=in_ch, base=32, num_classes=num_classes,
                                              latent_dim=16, input_length=kw.get('input_length', 1800))
    def forward(self, x): return self.core(x)


@_register_model
class TinySep1D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, **kw):
        super().__init__()
        self.core = TinySeparableCNN(in_ch, num_classes, **kw) if 'TinySeparableCNN' in globals() \
                    else SharedCoreSeparable1D(in_ch=in_ch, base=16, num_classes=num_classes,
                                               latent_dim=16, input_length=kw.get('input_length', 1800))
    def forward(self, x): return self.core(x)


@_register_model
class HypertinyHybrid(nn.Module):
    def __init__(self, dz=4, dh=12, in_ch=1, num_classes=2, base=16, latent_dim=16, input_length=1800, **kw):
        super().__init__()
        if 'build_hypertiny_hybrid' in globals():
            self.core = build_hypertiny_hybrid()
        elif 'TinyMethodModel' in globals():
            try:
                self.core = TinyMethodModel(in_ch, num_classes, keep_first_pw=True)
            except TypeError:
                self.core = TinyMethodModel(in_ch, num_classes)
        else:
            self.core = SharedCoreSeparable1D(in_ch=in_ch, base=base, num_classes=num_classes,
                                              latent_dim=latent_dim, input_length=input_length, hybrid_keep=1)
        self._synth_cfg = {"dz": dz, "dh": dh, "mode": "hybrid"}
    def forward(self, x): return self.core(x)


@_register_model
class HypertinyAllSynth(nn.Module):
    def __init__(self, dz=6, dh=16, in_ch=1, num_classes=2, base=16, latent_dim=16, input_length=1800, **kw):
        super().__init__()
        if 'build_hypertiny_all_synth' in globals():
            self.core = build_hypertiny_all_synth()
        elif 'TinyMethodModel' in globals():
            self.core = TinyMethodModel(in_ch, num_classes)
        else:
            self.core = SharedCoreSeparable1D(in_ch=in_ch, base=base, num_classes=num_classes,
                                              latent_dim=latent_dim, input_length=input_length)
        self._synth_cfg = {"dz": dz, "dh": dh, "mode": "all_synth"}
    def forward(self, x): return self.core(x)

# Minimal VAE encoder+head so TinyVAEHead exists if needed

@_register_model
class VAE1D_Enc(nn.Module):
    def __init__(self, in_ch=1, base=16, latent_dim=16, input_length=1800, **kw):
        super().__init__()
        assert 'TinyVAE1D' in globals(), "TinyVAE1D must exist for VAE1D_Enc"
        self.vae = TinyVAE1D(in_channels=in_ch, base=base, latent_dim=latent_dim, input_length=input_length)
        self.latent_dim = latent_dim
    @torch.no_grad()
    def encode(self, x):
        mu, logvar = self.vae.encode(x)
        return mu, logvar


@_register_model
class TinyVAEHead(nn.Module):
    """Simple VAE encoder + linear head so the name exists for both suites."""
    def __init__(self, in_ch=1, num_classes=2, z=16, base=16, input_length=1800, **kw):
        super().__init__()
        self.enc = VAE1D_Enc(in_ch=in_ch, base=base, latent_dim=z, input_length=input_length)
        self.head = nn.Linear(z, num_classes)
    def forward(self, x):
        mu, logvar = self.enc.encode(x)
        std = (0.5*logvar).exp().clamp_min(1e-3)
        z = mu + torch.randn_like(std) * std
        return self.head(z)
 #==================================Just Extra remove later
 #'''

@_register_model
class SafeFocalLoss(nn.Module):
    """
    Stable multi-class focal loss (supports hard labels or soft/one-hot).
    """
    def __init__(self, gamma=1.5, alpha=0.5, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C)
        if target.dtype in (torch.long, torch.int64):
            C = logits.size(1)
            with torch.no_grad():
                smooth = self.label_smoothing
                target_prob = torch.full_like(logits, smooth / max(1, C - 1))
                target_prob.scatter_(1, target.view(-1,1), 1.0 - smooth)
        else:
            target_prob = target  # already soft / one-hot

        logp = F.log_softmax(logits, dim=1)        # stable
        p = logp.exp()
        pt = (p * target_prob).sum(dim=1).clamp_min(1e-8)

        ce = -(target_prob * logp).sum(dim=1)      # smoothed CE
        focal = (self.alpha * (1.0 - pt).pow(self.gamma)) * ce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

@_register_model
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


@_register_model
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [N]
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1)
        loss = ((1-pt)**self.gamma) * ce
        if self.alpha is not None:
            alpha_vec = torch.ones(logits.size(1), device=logits.device)
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                alpha_vec = torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device)
            loss = alpha_vec[target] * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


@_register_model
class SqueezeExcite1D(nn.Module):
    """Lightweight SE block for 1D signals - improves feature selection"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        red = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(red, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


@_register_model
class DepthwiseSeparable1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, stride=1, padding=None, use_se=True, use_residual=True):
        super().__init__()
        padding = (k//2) if padding is None else padding
        self.use_residual = use_residual and (in_ch == out_ch) and (stride == 1)

        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # Optional squeeze-excitation
        self.se = SqueezeExcite1D(out_ch) if use_se else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        # Depthwise conv
        out = self.dw(x)
        out = self.bn1(out)
        out = self.act(out)

        # Pointwise conv
        out = self.pw(out)
        out = self.bn2(out)

        # SE block
        if self.se is not None:
            out = self.se(out)

        # Residual connection
        if self.use_residual:
            out = out + identity

        out = self.act(out)
        return out

@torch.no_grad()

@_register_model
class SharedPWGenerator(nn.Module):
    """Enhanced latent-to-weight generator with better expressivity"""
    def __init__(self, z_dim=16, hidden=64):
        super().__init__()
        self.z = nn.Parameter(torch.randn(z_dim) * 0.02)  # Even smaller init
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),  # Better than BatchNorm for small latents
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self):
        h = self.net(self.z)
        return h


@_register_model
class PWHead(nn.Module):
    """Enhanced projection with better weight generation"""
    def __init__(self, h_dim, flat_out):
        super().__init__()
        mid_dim = max(1, min(h_dim, flat_out // 2))
        self.proj = nn.Sequential(
            nn.Linear(h_dim, mid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(mid_dim, flat_out, bias=False)
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.05)

    def forward(self, h):
        return self.proj(h)

# ==== Channel Split Safety Helper ====

@_register_model
class MultiScaleFeatures(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Split out_ch across 3 branches while preserving the exact sum
        out_ch = _derive_out_ch(out_ch, in_ch)
        b1 = out_ch // 3
        b2 = out_ch // 3
        b3 = out_ch - (b1 + b2)  # absorbs remainder so b1+b2+b3 == out_ch
        assert b1 > 0 and b2 > 0 and b3 > 0, "out_ch must be >= 3"

        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, b1, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(in_ch, b2, kernel_size=5, padding=2, bias=False),
            nn.Conv1d(in_ch, b3, kernel_size=7, padding=3, bias=False),
        ])
        self.bn = nn.BatchNorm1d(out_ch)  # matches concat channels
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        # assert features.shape[1] == self.bn.num_features, f"BN expects {self.bn.num_features}, got {features.shape[1]}"
        return self.act(self.bn(features))


@_register_model
class SharedCoreSeparable1D(nn.Module):
    """
    Enhanced proposed model with:
    - Multi-scale feature extraction
    - Squeeze-Excitation attention
    - Residual connections
    - Learned attention weights + global pooling
    - Better classifier head
    """
    def __init__(self, in_ch=1, base=16, num_classes=2, latent_dim=16, input_length=1800, hybrid_keep=1):
        super().__init__()
        self.base = base

        # Stem with multi-scale features
        self.stem = nn.Sequential(
            MultiScaleFeatures(in_ch, base),
            nn.MaxPool1d(2, 2)  # Reduce temporal dimension
        )

        # Depthwise-separable stages
        self.blocks = nn.ModuleList([
            DepthwiseSeparable1D(base, base*2, k=5, stride=2, use_se=True, use_residual=False),
            DepthwiseSeparable1D(base*2, base*2, k=5, stride=1, use_se=True, use_residual=True),
            DepthwiseSeparable1D(base*2, base*4, k=5, stride=2, use_se=True, use_residual=False),
        ])

        # PW weight generator for the last pointwise conv (synthetic)
        self.gen = SharedPWGenerator(z_dim=latent_dim, hidden=96)
        self.last_pw_out = base*4
        self.last_pw_in  = base*2
        last_pw_shape = (self.last_pw_out, self.last_pw_in, 1)
        self.pw_head = PWHead(h_dim=96, flat_out=int(np.prod(last_pw_shape)))

        # Attention weight head: produces (B,1,T) weights in [0,1]
        self.att_weight = nn.Sequential(
            nn.Conv1d(base*4, max(1, base//4), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, base//4), 1, 1),
            nn.Sigmoid()
        )

        # Feature dim after pooling is channels of last stage
        feat_dim = base * 4

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _synth_pw_weight(self):
        h = self.gen()
        w = self.pw_head(h)
        w = w.view(self.last_pw_out, self.last_pw_in, 1)
        # Keep synthetic weights small
        w = torch.tanh(w) * 0.05
        return w

    def _forward_features(self, x):
        # --- NEW: sanitize + standardize inputs ---
        x = torch.nan_to_num(x)
        x = _standardize_1d(x)

        # light noise only during training (kept)
        if self.training:
            x = x + torch.randn_like(x) * 5e-7

        x = self.stem(x)
        x = torch.nan_to_num(x)

        # First two blocks
        x = self.blocks[0](x); x = torch.nan_to_num(x)
        x = self.blocks[1](x); x = torch.nan_to_num(x)

        # Third block: depthwise + BN/act, then synthetic PW, BN, SE, act
        b2 = self.blocks[2].dw(x)
        b2 = self.blocks[2].bn1(b2)
        b2 = self.blocks[2].act(b2)

        # Synthetic PW conv
        w = self._synth_pw_weight()
        b2 = F.conv1d(b2, w, bias=None, stride=1, padding=0, groups=1)
        b2 = self.blocks[2].bn2(b2)

        if self.blocks[2].se is not None:
            b2 = self.blocks[2].se(b2)
        b2 = self.blocks[2].act(b2)
        b2 = torch.nan_to_num(b2)

        # Attention-weighted global pooling (already stable with +1e-6, keep)
        att = self.att_weight(b2)               # (B,1,T)
        b2_weighted = b2 * att                  # (B,C,T)
        y = b2_weighted.sum(dim=-1) / (att.sum(dim=-1) + 1e-6)  # (B,C)

        # --- NEW: final sanitize ---
        y = torch.nan_to_num(y)
        return y

    def forward(self, x):
        y = self._forward_features(x)
        return self.head(y)

    def tinyml_packed_bytes(self):
        total = 0
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                total += (m.weight.numel() * 8 + 7)//8
        for p in list(self.gen.parameters()) + list(self.pw_head.parameters()):
            total += (p.numel() * 4 + 7)//8
        return {"boot": 1954, "lazy": total}

import torch
import torch.nn as nn
import torch.nn.functional as F

@_register_model
class SafeFocalLoss(nn.Module):
    """
    Stable multi-class focal loss (supports hard labels or soft/one-hot).
    """
    def __init__(self, gamma=1.5, alpha=0.5, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C)
        if target.dtype in (torch.long, torch.int64):
            C = logits.size(1)
            with torch.no_grad():
                smooth = self.label_smoothing
                target_prob = torch.full_like(logits, smooth / max(1, C - 1))
                target_prob.scatter_(1, target.view(-1,1), 1.0 - smooth)
        else:
            target_prob = target  # already soft / one-hot

        logp = F.log_softmax(logits, dim=1)        # stable
        p = logp.exp()
        pt = (p * target_prob).sum(dim=1).clamp_min(1e-8)

        ce = -(target_prob * logp).sum(dim=1)      # smoothed CE
        focal = (self.alpha * (1.0 - pt).pow(self.gamma)) * ce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


@_register_model
class TinyVAE1D(nn.Module):
    def __init__(self, in_channels=1, base=16, latent_dim=16, input_length=1800):
        super().__init__()
        self.latent_dim = latent_dim

        # Enhanced encoder with residual connections
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, base, 7, 2, 3), nn.BatchNorm1d(base), nn.ReLU(),
            DepthwiseSeparable1D(base, base*2, k=5, stride=2, use_se=False, use_residual=False),
            DepthwiseSeparable1D(base*2, base*2, k=5, stride=2, use_se=True, use_residual=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            e = self.enc(dummy)
            self._enc_flat = e.shape[1] * e.shape[2]
            self._enc_channels = e.shape[1]
            self._enc_length = e.shape[2]

        # Enhanced latent projection
        self.fc_mu = nn.Sequential(
            nn.Linear(self._enc_flat, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )
        self.fc_lv = nn.Sequential(
            nn.Linear(self._enc_flat, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, latent_dim)
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, self._enc_flat)
        )

        # Enhanced decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base*2, base*2, 4, 2, 1), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.ConvTranspose1d(base*2, base, 4, 2, 1), nn.BatchNorm1d(base), nn.ReLU(),
            nn.ConvTranspose1d(base, in_channels, 4, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        # --- NEW: sanitize + standardize before encode ---
        x = torch.nan_to_num(x)
        x = _standardize_1d(x)
        h = self.enc(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        lv = self.fc_lv(h)
        # tighter & safer clamps
        mu = torch.nan_to_num(mu).clamp(-10, 10)
        lv = torch.nan_to_num(lv).clamp(-8, 8)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv).clamp_min(1e-3)   # --- NEW: avoid near-zero std
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.dec_fc(z).view(z.size(0), self._enc_channels, self._enc_length)
        out = self.dec(h)
        # --- NEW: bound + sanitize decoder output to avoid exploding recon ---
        out = torch.tanh(out)                     # keep outputs finite
        return torch.nan_to_num(out)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        xhat = self.decode(z)
        return xhat, mu, lv


@_register_model
class VAEAdapter(nn.Module):
    """Enhanced adapter with feature refinement"""
    def __init__(self, vae: TinyVAE1D):
        super().__init__()
        self.vae = vae
        self.refine = nn.Sequential(
            nn.Linear(vae.latent_dim, vae.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(vae.latent_dim, vae.latent_dim)
        )

    def forward(self, x):
        mu, lv = self.vae.encode(x)
        refined = self.refine(mu)
        return refined + mu  # Residual connection


@_register_model
class AttentionPool1D(nn.Module):
    """
    Parameter-free temporal attention pooling.
    x: (B, C, T) -> returns (B, C)
    """
    def forward(self, x):
        score = x.mean(dim=1, keepdim=True)       # (B,1,T)
        alpha = torch.softmax(score, dim=-1)      # (B,1,T)
        return (x * alpha).sum(dim=-1)            # (B,C)


@_register_model
class TinyHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden*2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.net(z)


@_register_model
class SeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=k//2, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


@_register_model
class TinySeparableCNN(nn.Module):
    """Lightweight separable CNN for TinyML"""
    def __init__(self, in_ch, num_classes, base_filters=16, n_blocks=2):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.append(SeparableBlock(cur_ch, out_ch))
            if i < n_blocks - 1:  # no pooling after last block
                layers.append(nn.MaxPool1d(2))
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


@_register_model
class TinyVAEHead(nn.Module):
    """VAE encoder + linear head (no decoder for inference)"""
    def __init__(self, in_ch, num_classes, latent_dim=16, base_filters=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, base_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)  # reduce to manageable size
        self.fc_mu = nn.Linear(base_filters*2*8, latent_dim)
        self.fc_logvar = nn.Linear(base_filters*2*8, latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)  # (B, C*L)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.head(z)


@_register_model
class TinyMethodModel(nn.Module):
    """Prototype of the method: synthesis MLP for channel mixing"""
    def __init__(self, in_ch, num_classes, base_filters=16, latent_dim=8):
        super().__init__()
        # First layer: normal conv (would be kept in INT8)
        self.stem = nn.Conv1d(in_ch, base_filters, 3, padding=1)

        # Synthesis MLP (tiny)
        self.synthesis_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, base_filters * base_filters)  # for 1x1 conv weights
        )

        # Learnable latent code
        self.latent_code = nn.Parameter(torch.randn(latent_dim))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        # Apply stem
        x = F.relu(self.stem(x))  # (B, base_filters, L)

        # Synthesize pointwise conv weights
        synth_weights = self.synthesis_mlp(self.latent_code)  # (base_filters^2,)
        synth_weights = synth_weights.view(x.shape[1], x.shape[1], 1)  # (out, in, 1)

        # Apply synthesized conv
        x = F.conv1d(x, synth_weights)
        x = F.relu(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# -------------------- Quick test runner ----------------

@_register_model
class SeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=k//2, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


@_register_model
class TinySeparableCNN(nn.Module):
    """Lightweight separable CNN for TinyML - baseline model"""
    def __init__(self, in_ch, num_classes, base_filters=16, n_blocks=2):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.append(SeparableBlock(cur_ch, out_ch))
            if i < n_blocks - 1:  # no pooling after last block
                layers.append(nn.MaxPool1d(2))
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


@_register_model
class TinyVAEHead(nn.Module):
    """VAE encoder + linear head (no decoder for inference) - feature-forward baseline"""
    def __init__(self, in_ch, num_classes, latent_dim=16, base_filters=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, base_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)  # reduce to manageable size
        self.fc_mu = nn.Linear(base_filters*2*8, latent_dim)
        self.fc_logvar = nn.Linear(base_filters*2*8, latent_dim)
        self.head = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)  # (B, C*L)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.head(z)


@_register_model
class TinyMethodModel(nn.Module):
    """The Method: Generative compression with synthesis MLP for channel mixing"""
    def __init__(self, in_ch, num_classes, base_filters=16, latent_dim=8):
        super().__init__()
        # First layer: normal conv (would be kept in INT8 for stability)
        self.stem = nn.Conv1d(in_ch, base_filters, 3, padding=1)

        # Tiny synthesis MLP - replaces stored pointwise weights
        self.synthesis_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, base_filters * base_filters)  # synthesize 1x1 conv weights
        )

        # Learnable per-layer latent code (tiny storage)
        self.latent_code = nn.Parameter(torch.randn(latent_dim))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        # Apply stem (kept in INT8)
        x = F.relu(self.stem(x))  # (B, base_filters, L)

        # Synthesize pointwise conv weights from latent code
        synth_weights = self.synthesis_mlp(self.latent_code)  # (base_filters^2,)
        synth_weights = synth_weights.view(x.shape[1], x.shape[1], 1)  # (out, in, 1)

        # Apply synthesized conv (generated at boot, not stored)
        x = F.conv1d(x, synth_weights)
        x = F.relu(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)

@_register_model
class RegularCNN(nn.Module):
    """A regular CNN without TinyML constraints for comparison"""
    def __init__(self, input_length=1800, num_classes=2):
        super().__init__()
        self.input_length = input_length

        # Larger feature extractor for baseline comparison
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 5
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)         # (B, 512, 1)
        x = x.view(x.size(0), -1)    # (B, 512)
        x = self.classifier(x)       # (B, C)
        return x


@_register_model
class HRVFeatNet(nn.Module):
    """
    Computes a fixed 16D HRV(+amp) feature vector per window and learns a tiny linear head.
    Training only updates the linear layer; feature extraction is a deterministic transform.
    """
    def __init__(self, num_classes: int = 2, fs: float = 100.0):
        super().__init__()
        self.fs = float(fs)
        self.d  = 16
        self.head = nn.Linear(self.d, num_classes, bias=True)

    @torch.no_grad()
    def _batch_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        B = x.shape[0]
        feats = []
        for i in range(B):
            xi = x[i, 0].detach().cpu().numpy()
            fi = _hrv_features(xi, fs=self.fs)
            feats.append(fi)
        feats = torch.from_numpy(np.stack(feats, axis=0)).to(x.device)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self._batch_features(x)  # [B, 16]
        # normalize features lightly for stability
        f = (f - f.mean(dim=0, keepdim=True)) / (f.std(dim=0, keepdim=True) + 1e-6)
        return self.head(f)


# ---------------------------
# B) Compact 1D-CNN baseline
# ---------------------------

@_register_model
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, s=1, p=None, pool=2):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn   = nn.GroupNorm(1, c_out)
        self.act  = nn.SiLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=pool, stride=pool)

    def forward(self, x):
        x = self.pool(self.act(self.gn(self.conv(x))))
        return x


@_register_model
class CNN1D_3Blocks(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base=16):
        super().__init__()
        self.stem  = ConvBlock(in_ch, base, k=9, pool=2)
        self.b1    = ConvBlock(base, base*2, k=7, pool=2)
        self.b2    = ConvBlock(base*2, base*4, k=5, pool=2)
        self.head  = nn.Linear(base*4, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = x.mean(dim=-1)  # GAP
        return self.head(x)


# ---------------------------
# C) Tiny 1D-ResNet baseline
# ---------------------------

@_register_model
class BasicBlock1D(nn.Module):
    def __init__(self, c_in, c_out, stride=1, k=7):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c_in,  c_out, kernel_size=k, stride=stride, padding=p, bias=False)
        self.gn1   = nn.GroupNorm(1, c_out)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, stride=1,       padding=p, bias=False)
        self.gn2   = nn.GroupNorm(1, c_out)
        self.down  = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, c_out)
            )
    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.act(out + idt)
        return out


@_register_model
class ResNet1DSmall(nn.Module):
    """
    Stages: [base, 2*base, 2*base] with strides [2,2,2], 2 blocks per stage.
    ~O(50–80k) params for base=16.
    """
    def __init__(self, in_ch=1, num_classes=2, base=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(1, base),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            BasicBlock1D(base,     base,     stride=1),
            BasicBlock1D(base,     base,     stride=1),
        )
        self.stage2 = nn.Sequential(
            BasicBlock1D(base,     base*2,   stride=2),
            BasicBlock1D(base*2,   base*2,   stride=1),
        )
        self.stage3 = nn.Sequential(
            BasicBlock1D(base*2,   base*2,   stride=2),
            BasicBlock1D(base*2,   base*2,   stride=1),
        )
        self.head = nn.Linear(base*2, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.mean(dim=-1)  # GAP
        return self.head(x)


# === Model registry ===
MODEL_BUILDERS = {}

def _register_model(cls):
    """Decorator to add a model class to MODEL_BUILDERS by name (lowercase)."""
    name = cls.__name__
    key = name.lower()
    # default builder that forwards kwargs but filters unknown args if constructor is strict
    def builder(in_ch: int, num_classes: int, **kwargs):
        import inspect
        sig = inspect.signature(cls.__init__)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_varkw:
            return cls(in_ch, num_classes, **kwargs)
        # keep only ctor-supported kwargs
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        try:
            return cls(in_ch, num_classes, **filtered)
        except TypeError:
            # greedy inclusion
            accepted = {}
            for k, v in filtered.items():
                try:
                    _ = cls(in_ch, num_classes, **accepted, **{k: v})
                    accepted[k] = v
                except TypeError:
                    continue
            return cls(in_ch, num_classes, **accepted)

    # prefer last-writer-wins to allow overrides
    MODEL_BUILDERS[key] = builder
    return cls

def safe_build_model(model_name: str, in_ch: int, num_classes: int, **model_kwargs):
    key = model_name.lower()
    if key not in MODEL_BUILDERS:
        raise KeyError(f"Model '{model_name}' not registered. Available: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[key](in_ch, num_classes, **model_kwargs)
