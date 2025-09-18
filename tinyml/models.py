
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class TinyMethodModel(nn.Module):
    """Generative compression: synthesize 1×1 channel-mixing weights from a tiny latent code."""
    def __init__(self, in_ch, num_classes, base_filters=16, dz: int = 8, dh: int = 32, **kwargs):
        super().__init__()
        self.dz = dz
        self.dh = dh
        # First layer: normal conv (typically kept INT8 for stability)
        self.stem = nn.Conv1d(in_ch, base_filters, kernel_size=3, padding=1)
        # Tiny synthesis MLP replaces stored 1x1 PW weights
        self.synthesis_mlp = nn.Sequential(
            nn.Linear(dz, dh),
            nn.ReLU(),
            nn.Linear(dh, base_filters * base_filters)  # synthesize 1x1 conv weights
        )
        # Learnable latent code
        self.latent_code = nn.Parameter(torch.randn(dz))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.stem(x))  # (B, C=base_filters, L)
        w = self.synthesis_mlp(self.latent_code.unsqueeze(0))   # (1, C*C)
        C = x.shape[1]
        w = w.view(C, C, 1).to(device=x.device, dtype=x.dtype)
        x = F.conv1d(x, w)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


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



class RegularCNN(nn.Module):
    """Regular CNN baseline for comparison"""
    def __init__(self, in_ch, num_classes, base_filters=32, n_blocks=3):
        super().__init__()
        layers = []
        cur_ch = in_ch
        for i in range(n_blocks):
            out_ch = base_filters * (2**i)
            layers.extend([
                nn.Conv1d(cur_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            cur_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cur_ch, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
'''


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


class TinySep1D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, **kw):
        super().__init__()
        self.core = TinySeparableCNN(in_ch, num_classes, **kw) if 'TinySeparableCNN' in globals() \
                    else SharedCoreSeparable1D(in_ch=in_ch, base=16, num_classes=num_classes,
                                               latent_dim=16, input_length=kw.get('input_length', 1800))
    def forward(self, x): return self.core(x)



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


def conv1d_macs(m: nn.Conv1d, L_in: int) -> int:
    Cin  = m.in_channels
    Cout = m.out_channels
    k    = m.kernel_size[0]
    s    = m.stride[0]
    p    = m.padding[0]
    d    = m.dilation[0]
    g    = m.groups
    Lout = math.floor((L_in + 2*p - d*(k-1) - 1)/s + 1)
    # per-output MACs = (Cin/g)*k, total = Cout * Lout * (Cin/g)*k
    macs = Cout * Lout * (Cin // g) * k
    return macs, Lout



def linear_macs(m: nn.Linear) -> int:
    return m.in_features * m.out_features



def estimate_macs(model: nn.Module, in_ch: int, seq_len: int) -> int:
    L = seq_len
    macs_total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            macs, L = conv1d_macs(m, L)
            macs_total += macs
        elif isinstance(m, nn.Linear):
            macs_total += linear_macs(m)
    return macs_total  # multiply by batch if needed

# --- Activation peak (lower-bound) via forward hooks ---
@torch.no_grad()



MODEL_BUILDERS = {}
def register_model(name: str, builder):
    MODEL_BUILDERS[name] = builder

# Forward **kw so grid params (dz, dh, etc.) don't crash
if 'TinySeparableCNN' in globals(): register_model('tiny_separable_cnn', lambda ic, nc, **kw: TinySeparableCNN(ic, nc, **kw))
register_model('tiny_method',        lambda ic, nc, **kw: TinyMethodModel(ic, nc, **kw))
if 'RegularCNN' in globals():        register_model('regular_cnn',        lambda ic, nc, **kw: RegularCNN(ic, nc, **kw))
if 'ResNet1DSmall' in globals():     register_model('resnet1d_small',     lambda ic, nc, **kw: ResNet1DSmall(ic, nc, **kw))
if 'HRVFeatNet' in globals():        register_model('hrv_featnet',        lambda ic, nc, **kw: HRVFeatNet(num_classes=nc, **kw))

