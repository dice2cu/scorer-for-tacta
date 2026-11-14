"""Tiny fully-convolutional detector for Tacta badges and dots (CenterNet-style, stride 4).

Runtime budget (phones, in-browser, the app's own WebGL2/CPU engine): < 1M params, standard ops only
(Conv, BatchNorm folded at export, ReLU, Add, Resize-nearest, Sigmoid). Input: RGB, any size
divisible by 16, scaled so a card is ~100 px wide.

Output channels at stride 4 (see HEADS):
  badge_hm[4]   circle, square, triangle, start-card centre heatmaps (logits)
  badge_off[2]  sub-cell offset of the badge centre
  value[6]      card value 1..6 (logits)
  side[1]       back side (logit)
  deck[6]       deck colour of the card (logits)
  icon[2]       vector badge -> icon, in px / ICON_SCALE
  dot_hm[1]     visible-dot centre heatmap (logit)
  dot_off[2]    sub-cell offset of the dot centre
  dot_deck[6]   deck colour (= owner) of the visible dot (logits)
  dot_part[1]   partly covered dot (a misaligned card cuts it): not counted, flagged (logit)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DECKS = ["blue", "green", "orange", "pink", "purple", "red"]
SUITS = ["circle", "square", "triangle", "start"]
HEADS = [("badge_hm", 4), ("badge_off", 2), ("value", 6), ("side", 1), ("deck", 6), ("icon", 2),
         ("dot_hm", 1), ("dot_off", 2), ("dot_deck", 6),
         ("dot_part", 1)]
N_OUT = sum(n for _, n in HEADS)
STRIDE = 4
ICON_SCALE = 32.0


def head_slices():
    out, k = {}, 0
    for name, n in HEADS:
        out[name] = slice(k, k + n)
        k += n
    return out


def conv_bn(cin, cout, k=3, s=1, groups=1, dilation=1, act=True):
    pad = dilation * (k // 2)
    layers = [nn.Conv2d(cin, cout, k, s, pad, dilation=dilation, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DWBlock(nn.Module):
    """Depthwise-separable residual block (MobileNet-style, expansion 2)."""

    def __init__(self, cin, cout, stride=1, dilation=1, expand=2):
        super().__init__()
        mid = cin * expand
        self.body = nn.Sequential(
            conv_bn(cin, mid, k=1),
            conv_bn(mid, mid, k=3, s=stride, groups=mid, dilation=dilation),
            conv_bn(mid, cout, k=1, act=False),
        )
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return F.relu(x + y) if self.res else F.relu(y)


class TactaNet(nn.Module):
    def __init__(self, w=(16, 32, 56, 96), head_ch=40):
        super().__init__()
        c1, c2, c3, c4 = w
        self.stem = conv_bn(3, c1, s=2)                                    # /2
        self.s4 = nn.Sequential(DWBlock(c1, c2, 2, expand=2), DWBlock(c2, c2, expand=1))   # /4
        self.s8 = nn.Sequential(DWBlock(c2, c3, 2), DWBlock(c3, c3))                       # /8
        self.s16 = nn.Sequential(DWBlock(c3, c4, 2), DWBlock(c4, c4), DWBlock(c4, c4, dilation=2),
                                 DWBlock(c4, c4, dilation=4))               # /16, wide context
        self.lat8 = conv_bn(c4, c3, k=1)
        self.fuse8 = DWBlock(c3, c3, expand=1)
        self.lat4 = conv_bn(c3, c2, k=1)
        self.fuse4 = DWBlock(c2, c2, expand=1)
        # cheap high-resolution head: 1x1 -> depthwise 3x3 -> 1x1 outputs
        self.head = nn.Sequential(conv_bn(c2, head_ch, k=1), conv_bn(head_ch, head_ch, k=3, groups=head_ch),
                                  nn.Conv2d(head_ch, N_OUT, 1))
        # heatmap logits start low (CenterNet prior)
        sl = head_slices()
        with torch.no_grad():
            self.head[-1].bias[sl["badge_hm"]] = -4.0
            self.head[-1].bias[sl["dot_hm"]] = -4.0
            self.head[-1].bias[sl["dot_part"]] = -4.0

    def forward(self, x):
        x = self.stem(x)
        f4 = self.s4(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        u8 = self.fuse8(f8 + F.interpolate(self.lat8(f16), scale_factor=2, mode="nearest"))
        u4 = self.fuse4(f4 + F.interpolate(self.lat4(u8), scale_factor=2, mode="nearest"))
        return self.head(u4)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_model(path, device="cpu"):
    """Load a checkpoint into the current TactaNet. Checkpoints from before an output channel was
    added still load: the missing channels get a strongly negative bias (they never fire)."""
    model = TactaNet().to(device).eval()
    src = torch.load(path, map_location=device, weights_only=False)["model"]
    own = model.state_dict()
    for k, v in src.items():
        if own[k].shape == v.shape:
            own[k] = v
        else:
            own[k][: v.shape[0]] = v
            own[k][v.shape[0]:] = 0 if own[k].dim() > 1 else -20.0
    model.load_state_dict(own)
    return model


if __name__ == "__main__":
    m = TactaNet()
    print(f"params: {count_params(m):,}")
    x = torch.zeros(1, 3, 384, 384)
    print("out:", tuple(m(x).shape))

