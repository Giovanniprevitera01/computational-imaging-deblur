"""
NAF-Net semplificato e corretto.
"""

import torch
import torch.nn as nn


# -------------------------------------------------
# Simple Gate
# -------------------------------------------------
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# -------------------------------------------------
# NAF Block
# -------------------------------------------------
class NAFBlock(nn.Module):

    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()

        dw_c = c * dw_expand

        # ---------------- Attention branch ----------------

        self.conv1 = nn.Conv2d(c, dw_c, 1)

        self.conv2 = nn.Conv2d(
            dw_c,
            dw_c,
            kernel_size=3,
            padding=1,
            groups=dw_c
        )

        self.sg = SimpleGate()

        # dopo SimpleGate: dw_c -> dw_c // 2
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_c // 2, dw_c // 2, 1),
            nn.Sigmoid()
        )

        self.conv3 = nn.Conv2d(dw_c // 2, c, 1)

        # ---------------- FFN branch ----------------

        ffn_c = c * ffn_expand

        self.ff1 = nn.Conv2d(c, ffn_c, 1)
        self.ff2 = nn.Conv2d(ffn_c // 2, c, 1)

        # residual scaling
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):

        # -------- attention branch --------
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.ca(x)

        x = self.conv3(x)

        y = inp + x * self.beta

        # -------- FFN branch --------
        x = self.ff1(y)
        x = self.sg(x)
        x = self.ff2(x)

        return y + x * self.gamma


# -------------------------------------------------
# NAFNet
# -------------------------------------------------
class NAFNet(nn.Module):

    def __init__(
        self,
        inp_channels=4,
        out_channels=3,
        width=32,
        enc_blks=[1, 1, 1],
        dec_blks=[1, 1, 1]
    ):

        super().__init__()

        self.intro = nn.Conv2d(
            inp_channels,
            width,
            kernel_size=3,
            padding=1
        )

        self.ending = nn.Conv2d(
            width,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.middle = nn.Sequential(
            *[NAFBlock(width * 8) for _ in range(4)]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = width

        # ---------------- Encoder ----------------
        for n in enc_blks:

            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(ch) for _ in range(n)]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    ch,
                    ch * 2,
                    kernel_size=2,
                    stride=2
                )
            )

            ch *= 2

        # middle channels
        middle_ch = ch

        # ---------------- Decoder ----------------
        for n in dec_blks:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(middle_ch, middle_ch * 2, 1),
                    nn.PixelShuffle(2)
                )
            )

            middle_ch //= 2

            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(middle_ch) for _ in range(n)]
                )
            )

    def forward(self, inp, sigma):

        B, _, H, W = inp.shape

        # sigma conditioning
        sigma_map = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)

        x = torch.cat([inp, sigma_map], dim=1)

        x = self.intro(x)

        skips = []

        # ---------------- Encoder ----------------
        for enc, down in zip(self.encoders, self.downs):

            x = enc(x)

            skips.append(x)

            x = down(x)

        # ---------------- Middle ----------------
        x = self.middle(x)

        # ---------------- Decoder ----------------
        for dec, up, skip in zip(
            self.decoders,
            self.ups,
            reversed(skips)
        ):

            x = up(x)

            x = x + skip

            x = dec(x)

        # residual learning
        return self.ending(x) + inp
