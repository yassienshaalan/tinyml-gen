import math, os, json, random
from typing import Any, Dict, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Model Registry (define FIRST so decorators exist)
# ============================================================

MODEL_BUILDERS: Dict[str, Callable] = {}
MODEL_ALIASES: Dict[str, str] = {}


import inspect  # <-- add this near the top with other imports

def _register_model(name: Optional[str] = None):
    """
    Decorator for nn.Module classes. Registers a builder that:
      - maps common alias kwargs (dz->latent_dim, qbits->quant_bits)
      - filters unknown kwargs so class __init__ doesn't error
      - tries handy ctor signatures
    """
    def _wrap(cls):
        key = (name or cls.__name__).lower()
        if key in MODEL_BUILDERS:
            raise ValueError(f"Duplicate model registration for '{key}'")

        def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            # Map generic grid kwargs → class-specific names
            mapping = {
                "dz": "latent_dim",     # your method uses a single latent size
                "qbits": "quant_bits",  # common alias
            }
            kw = {mapping.get(k, k): v for k, v in kwargs.items()}
            # Keep only kwargs that the target class __init__ accepts
            sig = inspect.signature(cls.__init__)
            allowed = {p.name for p in sig.parameters.values()}
            return {k: v for k, v in kw.items() if k in allowed}

        def _builder(in_ch: int, num_classes: int, **kwargs):
            kw = _filter_kwargs(kwargs)
            try:
                return cls(in_ch=in_ch, num_classes=num_classes, **kw)
            except TypeError:
                try:
                    return cls(num_classes=num_classes, **kw)
                except TypeError:
                    return cls(**kw)

        MODEL_BUILDERS[key] = _builder
        return cls
    return _wrap


def register_alias(alias: str, target: str):
    """Map a friendly alias to a registered key (both lower-cased)."""
    MODEL_ALIASES[alias.lower()] = target.lower()


def safe_build_model(model_name_or_cfg, in_ch: int = None, num_classes: int = None, **model_kwargs):
    """
    Resolve alias -> key, look up builder, and instantiate.
    Raises a clear error listing available names if not found.

    Accepts either positional args ``(name, in_ch, num_classes, **kw)``
    or a single config dict with at least a ``'name'`` key.
    """
    if isinstance(model_name_or_cfg, dict):
        cfg = dict(model_name_or_cfg)
        model_name = cfg.pop('name')
        in_ch = cfg.pop('in_channels', cfg.pop('in_ch', in_ch or 1))
        num_classes = cfg.pop('num_classes', num_classes or 2)
        model_kwargs.update(cfg)
    else:
        model_name = model_name_or_cfg

    key = MODEL_ALIASES.get(model_name.lower(), model_name.lower())
    if key not in MODEL_BUILDERS:
        avail = sorted(MODEL_BUILDERS.keys())
        raise KeyError(f"Model '{model_name}' not registered. Available: {avail}")
    return MODEL_BUILDERS[key](in_ch, num_classes, **model_kwargs)


# ============================================================
# Small Utilities used by multiple models
# ============================================================

def _standardize_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Per-sample, per-channel standardization over time dim.
    x: (B,C,T)
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _derive_out_ch(out_ch: int, in_ch: int) -> int:
    """
    Ensure out_ch is positive; simple safety wrapper (kept for API parity).
    """
    out_ch = int(out_ch)
    if out_ch <= 0:
        out_ch = max(1, in_ch)
    return out_ch


# Very lightweight placeholder for HRV-style features (16-dim)
# You can replace with your precise feature set later; this keeps shapes stable.
def _hrv_features(sig: np.ndarray, fs: float = 100.0) -> np.ndarray:
    x = np.asarray(sig).astype(np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    x = np.nan_to_num(x)
    L = max(1, x.size)
    t = np.arange(L, dtype=np.float32) / max(1.0, fs)

    feats = []
    # Stats
    feats += [x.mean(), x.std(), x.min(), x.max()]
    # Simple morphology
    feats += [np.median(x), np.percentile(x, 25), np.percentile(x, 75), np.ptp(x)]
    # Rough slope/energy
    dx = np.diff(x, prepend=x[:1])
    feats += [float(np.mean(np.abs(dx))), float(np.mean(dx**2))]
    # Very rough freq proxy
    X = np.fft.rfft(x)
    ps = np.abs(X)**2
    feats += [float(ps.mean()), float(ps.max())]
    # Duration + zcr-ish
    zc = np.mean((x[1:] * x[:-1]) < 0.0) if L > 1 else 0.0
    feats += [float(t[-1] if L > 0 else 0.0), float(zc)]

    feats = np.array(feats, dtype=np.float32)
    if feats.shape[0] < 16:
        feats = np.pad(feats, (0, 16 - feats.shape[0]), mode="constant")
    else:
        feats = feats[:16]
    return feats


# ============================================================
# Building Blocks
# ============================================================

@_register_model()
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


@_register_model()
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
        out = self.dw(x); out = self.bn1(out); out = self.act(out)
        out = self.pw(out); out = self.bn2(out)
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            out = out + identity
        return self.act(out)


class MultiScaleFeatures(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        out_ch = _derive_out_ch(out_ch, in_ch)
        b1 = out_ch // 3
        b2 = out_ch // 3
        b3 = out_ch - (b1 + b2)
        assert b1 > 0 and b2 > 0 and b3 > 0, "out_ch must be >= 3"

        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, b1, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(in_ch, b2, kernel_size=5, padding=2, bias=False),
            nn.Conv1d(in_ch, b3, kernel_size=7, padding=3, bias=False),
        ])
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.act(self.bn(features))


class SharedPWGenerator(nn.Module):
    """Latent-to-weight generator"""
    def __init__(self, z_dim=16, hidden=64):
        super().__init__()
        self.z = nn.Parameter(torch.randn(z_dim) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self):
        return self.net(self.z)


class PWHead(nn.Module):
    """Projection to conv(1x1) weights"""
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


# ============================================================
# Core Architectures
# ============================================================
def _fake_quant_per_tensor(x: torch.Tensor, nbits: int) -> torch.Tensor:
    """Symmetric fake-quant in training; pass-through in eval."""
    if nbits is None or nbits >= 32: 
        return x
    # avoid degenerate ranges
    xmax = x.detach().abs().amax()
    scale = (xmax / (2**(nbits-1)-1)).clamp(min=1e-8)
    q = torch.clamp(torch.round(x / scale), min=-(2**(nbits-1)), max=(2**(nbits-1)-1))
    return q * scale
	
@_register_model()
class SharedCoreSeparable1D(nn.Module):
    """
    TinyMethod (ours): multi-scale stem + depthwise blocks; final PW is synthesized
    from a compact latent (generator). Adds optional QAT fake-quant for features & PW.
    """
    def __init__(self, in_ch=1, base=16, num_classes=2, latent_dim=16, input_length=1800,
                 hybrid_keep=1,  # kept for API parity
                 qat_bits: int | None = None):  # <-- NEW: can be set at build or via set_qat()
        super().__init__()
        self.base = base
        self.qat_bits = qat_bits  # if None, disabled until set_qat() is called

        # Stem
        self.stem = nn.Sequential(
            MultiScaleFeatures(in_ch, base),
            nn.MaxPool1d(2, 2)
        )

        # Two regular DS blocks + one where PW is synthesized
        self.blocks = nn.ModuleList([
            DepthwiseSeparable1D(base,   base*2, k=5, stride=2, use_se=True,  use_residual=False),
            DepthwiseSeparable1D(base*2, base*2, k=5, stride=1, use_se=True,  use_residual=True),
            DepthwiseSeparable1D(base*2, base*4, k=5, stride=2, use_se=True,  use_residual=False),
        ])

        # Latent → weight generator for the last PW conv
        self.gen = SharedPWGenerator(z_dim=latent_dim, hidden=96)
        self.last_pw_out = base*4
        self.last_pw_in  = base*2
        self.pw_head = PWHead(h_dim=96, flat_out=int(self.last_pw_out*self.last_pw_in*1))

        # Lightweight attention pooling
        self.att_weight = nn.Sequential(
            nn.Conv1d(base*4, max(1, base//4), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, base//4), 1, 1),
            nn.Sigmoid()
        )

        feat_dim = base * 4
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),      nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # --- QAT helpers ---
    def set_qat(self, nbits: int):     self.qat_bits = int(nbits)
    def clear_qat(self):               self.qat_bits = None

    def _synth_pw_weight(self):
        h = self.gen()
        w = self.pw_head(h).view(self.last_pw_out, self.last_pw_in, 1)
        # keep small magnitude for stability
        w = torch.tanh(w) * 0.05
        # fake-quant PW kernel if QAT is on
        if self.training and self.qat_bits is not None:
            w = _fake_quant_per_tensor(w, self.qat_bits)
        return w

    def _forward_features(self, x):
        # sanitize + standardize
        x = torch.nan_to_num(x)
        x = _standardize_1d(x)
        if self.training:
            x = x + torch.randn_like(x) * 5e-7  # tiny noise for robustness

        # stem + first two DS blocks
        x = self.stem(x);           x = torch.nan_to_num(x)
        x = self.blocks[0](x);      x = torch.nan_to_num(x)
        x = self.blocks[1](x);      x = torch.nan_to_num(x)

        # third block with synthesized PW
        b2 = self.blocks[2].dw(x)
        b2 = self.blocks[2].bn1(b2)
        b2 = self.blocks[2].act(b2)

        w = self._synth_pw_weight()
        b2 = F.conv1d(b2, w, bias=None, stride=1, padding=0, groups=1)
        b2 = self.blocks[2].bn2(b2)
        if self.blocks[2].se is not None: b2 = self.blocks[2].se(b2)
        b2 = self.blocks[2].act(b2)

        # attention pooling
        att = self.att_weight(b2)                         # (B,1,T)
        y   = (b2 * att).sum(dim=-1) / (att.sum(dim=-1) + 1e-6)  # (B,C)

        # fake-quant pooled features if QAT is on
        if self.training and self.qat_bits is not None:
            y = _fake_quant_per_tensor(y, self.qat_bits)
        return torch.nan_to_num(y)

    def forward(self, x):
        y = self._forward_features(x)
        return self.head(y)


@_register_model()
class TinyVAE1D(nn.Module):
    def __init__(self, in_channels=1, base=16, latent_dim=16, input_length=1800):
        super().__init__()
        self.latent_dim = latent_dim

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

        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base*2, base*2, 4, 2, 1), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.ConvTranspose1d(base*2, base, 4, 2, 1),   nn.BatchNorm1d(base),   nn.ReLU(),
            nn.ConvTranspose1d(base, in_channels, 4, 2, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = torch.nan_to_num(x)
        x = _standardize_1d(x)
        h = self.enc(x).view(x.size(0), -1)
        mu = torch.nan_to_num(self.fc_mu(h)).clamp(-10, 10)
        lv = torch.nan_to_num(self.fc_lv(h)).clamp(-8, 8)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv).clamp_min(1e-3)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.dec_fc(z).view(z.size(0), self._enc_channels, self._enc_length)
        out = self.dec(h)
        return torch.nan_to_num(torch.tanh(out))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        xhat = self.decode(z)
        return xhat, mu, lv


@_register_model()
class VAEAdapter(nn.Module):
    """Adapter over TinyVAE1D → returns refined latent."""
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
        return refined + mu


@_register_model()
class AttentionPool1D(nn.Module):
    """Parameter-free temporal attention pooling. x: (B,C,T) -> (B,C)"""
    def forward(self, x):
        score = x.mean(dim=1, keepdim=True)       # (B,1,T)
        alpha = torch.softmax(score, dim=-1)      # (B,1,T)
        return (x * alpha).sum(dim=-1)            # (B,C)


@_register_model()
class TinyHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden*2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.net(z)


# ============================================================
# Baselines + Proposed Variants
# ============================================================

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


@_register_model()
class TinySeparableCNN(nn.Module):
    """Lightweight separable CNN for TinyML - baseline model"""
    def __init__(self, in_ch, num_classes, base_filters=16, n_blocks=2):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.append(SeparableBlock(cur_ch, out_ch))
            if i < n_blocks - 1:
                layers.append(nn.MaxPool1d(2))
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


@_register_model()
class TinyMethodModel(nn.Module):
    """Prototype of the method: synthesis MLP for channel mixing"""
    def __init__(self, in_ch, num_classes, base_filters=16, latent_dim=8):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base_filters, 3, padding=1)
        self.synthesis_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, base_filters * base_filters)
        )
        self.latent_code = nn.Parameter(torch.randn(latent_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.stem(x))  # (B, base_filters, L)
        synth_weights = self.synthesis_mlp(self.latent_code).view(x.shape[1], x.shape[1], 1)
        x = F.relu(F.conv1d(x, synth_weights))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


@_register_model()
class RegularCNN(nn.Module):
    """A regular CNN without TinyML constraints for comparison"""
    def __init__(self, input_length=1800, num_classes=2):
        super().__init__()
        self.input_length = input_length
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64,128,kernel_size=5,padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128,256,kernel_size=3,padding=1),nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(256,512,kernel_size=3,padding=1),nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(512,512,kernel_size=3,padding=1),nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)         # (B, 512, 1)
        x = x.view(x.size(0), -1)    # (B, 512)
        return self.classifier(x)

@_register_model()		
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
		
@_register_model()
class RegularCNN1D(nn.Module):
    """
    Wrapper that accepts (in_ch, num_classes) and builds RegularCNN.
    Keeps runner happy when it expects those params.
    """
    def __init__(self, in_ch=1, num_classes=2, **kw):
        super().__init__()
        self.core = RegularCNN(input_length=kw.get('input_length', 1800), num_classes=num_classes)
    def forward(self, x): return self.core(x)


@_register_model()
class HRVFeatNet(nn.Module):
    """
    Computes a fixed 16D HRV(+amp) feature vector per window and learns a tiny linear head.
    Training only updates the linear layer; feature extraction is deterministic.
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
        f = (f - f.mean(dim=0, keepdim=True)) / (f.std(dim=0, keepdim=True) + 1e-6)
        return self.head(f)


# ---------------------------
# Compact CNN / ResNet baselines
# ---------------------------

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
        return self.pool(self.act(self.gn(self.conv(x))))


@_register_model()
class CNN1D_3Blocks(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base=16):
        super().__init__()
        self.stem  = ConvBlock(in_ch, base, k=9, pool=2)
        self.b1    = ConvBlock(base, base*2, k=7, pool=2)
        self.b2    = ConvBlock(base*2, base*4, k=5, pool=2)
        self.head  = nn.Linear(base*4, num_classes)
    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x)
        x = x.mean(dim=-1)  # GAP
        return self.head(x)


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
        return self.act(out + idt)


@_register_model()
class ResNet1DSmall(nn.Module):
    """Stages: [base, 2*base, 2*base] with strides [2,2,2], 2 blocks per stage."""
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


# ============================================================
# Losses (kept for compatibility)
# ============================================================

class SafeFocalLoss(nn.Module):
    """Stable multi-class focal loss (supports hard labels or soft/one-hot)."""
    def __init__(self, gamma=1.5, alpha=0.5, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        if target.dtype in (torch.long, torch.int64):
            C = logits.size(1)
            with torch.no_grad():
                smooth = self.label_smoothing
                target_prob = torch.full_like(logits, smooth / max(1, C - 1))
                target_prob.scatter_(1, target.view(-1,1), 1.0 - smooth)
        else:
            target_prob = target

        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = (p * target_prob).sum(dim=1).clamp_min(1e-8)
        ce = -(target_prob * logp).sum(dim=1)
        focal = (self.alpha * (1.0 - pt).pow(self.gamma)) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


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


# ============================================================
# ============================================================
# HyperTinyPW — public API wrapper around SharedCoreSeparable1D
# ============================================================

class HyperTinyPW(SharedCoreSeparable1D):
    """
    Public-facing name for the HyperTiny pointwise-synthesis model.
    Maps the documented constructor interface to SharedCoreSeparable1D.
    """
    def __init__(self, num_classes=2, in_channels=1, base_channels=16,
                 num_blocks=4, latent_dim=16, seq_len=1800, **kwargs):
        super().__init__(
            in_ch=in_channels,
            base=base_channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            input_length=seq_len,
            **kwargs,
        )


# Aliases for runner configs (IMPORTANT)
# ============================================================

# names used by experiments/configs → registered class keys
register_alias("tiny_method",        "sharedcoreseparable1d")  # or "tinymethodmodel" if you prefer that core
register_alias("tiny_vae_head",      "tinyvaehead")            # wrapper below
register_alias("cnn3_small",         "cnn1d_3blocks")
register_alias("resnet1d_small",     "resnet1dsmall")
register_alias("tiny_separable_cnn", "tinyseparablecnn")
register_alias("regular_cnn",        "regularcnn1d")
register_alias("hrv_featnet",        "hrvfeatnet")
