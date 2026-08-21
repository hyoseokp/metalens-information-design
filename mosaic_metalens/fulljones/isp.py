from __future__ import annotations

import torch
import torch.nn.functional as F


def demosaic_bilinear(raw: torch.Tensor, bayer_masks: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=raw.device,
        dtype=raw.dtype,
    ) / 4.0
    kernel = kernel.view(1, 1, 3, 3)

    rgb = []
    raw_4d = raw.view(1, 1, *raw.shape)
    for channel in range(3):
        mask = bayer_masks[channel].view(1, 1, *bayer_masks.shape[-2:]).to(raw.dtype)
        values = F.conv2d(raw_4d * mask, kernel, padding=1)
        weights = F.conv2d(mask, kernel, padding=1).clamp_min(1e-6)
        rgb.append((values / weights).squeeze(0).squeeze(0))
    return torch.stack(rgb, dim=0)


def demosaic_malvar(raw: torch.Tensor, bayer_masks: torch.Tensor) -> torch.Tensor:
    device = raw.device
    dtype = raw.dtype
    height, width = raw.shape

    Kg = torch.tensor(
        [[0.0, 0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0,  2.0, 0.0, 0.0],
         [-1.0, 2.0, 4.0, 2.0, -1.0],
         [0.0, 0.0,  2.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    Kr_b = torch.tensor(
        [[0.0, 0.0, -1.5, 0.0, 0.0],
         [0.0, 2.0,  0.0, 2.0, 0.0],
         [-1.5, 0.0, 6.0, 0.0, -1.5],
         [0.0, 2.0,  0.0, 2.0, 0.0],
         [0.0, 0.0, -1.5, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    Kr_gr = torch.tensor(
        [[0.0, 0.0, 0.5, 0.0, 0.0],
         [0.0, -1.0, 0.0, -1.0, 0.0],
         [-1.0, 4.0, 5.0, 4.0, -1.0],
         [0.0, -1.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.5, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    Kr_gb = torch.tensor(
        [[0.0, 0.0, -1.0, 0.0, 0.0],
         [0.0, -1.0, 4.0, -1.0, 0.0],
         [0.5, 0.0, 5.0, 0.0, 0.5],
         [0.0, -1.0, 4.0, -1.0, 0.0],
         [0.0, 0.0, -1.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    ) / 8.0

    def conv5(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        image4 = image.view(1, 1, height, width)
        padded = F.pad(image4, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, kernel.view(1, 1, 5, 5)).squeeze(0).squeeze(0)

    r_mask = bayer_masks[0].to(dtype)
    g_mask = bayer_masks[1].to(dtype)
    b_mask = bayer_masks[2].to(dtype)

    yy = torch.arange(height, device=device)
    xx = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    # RGGB: R at (even,even), B at (odd,odd) ->
    #   G-in-R-row (horizontal R neighbors): even row, odd col
    #   G-in-B-row (vertical R neighbors):   odd row, even col
    gr_mask = ((yy % 2 == 0) & (xx % 2 == 1)).to(dtype)
    gb_mask = ((yy % 2 == 1) & (xx % 2 == 0)).to(dtype)

    green_interp = conv5(raw, Kg)
    green = raw * g_mask + green_interp * (1.0 - g_mask)

    red_gr = conv5(raw, Kr_gr)
    red_gb = conv5(raw, Kr_gb)
    red_b = conv5(raw, Kr_b)
    red = raw * r_mask + red_gr * gr_mask + red_gb * gb_mask + red_b * b_mask

    # at a gr site (R-row) the B neighbors are VERTICAL -> Kr_gr.T (vertical 4s);
    # at a gb site (B-row) the B neighbors are HORIZONTAL -> Kr_gb.T
    blue_gr = conv5(raw, Kr_gr.transpose(0, 1))
    blue_gb = conv5(raw, Kr_gb.transpose(0, 1))
    blue_r = red_b  # same kernel Kr_b on the same raw — reuse (bit-exact)
    blue = raw * b_mask + blue_gr * gr_mask + blue_gb * gb_mask + blue_r * r_mask

    # Keep the ISP linear. Callers that operate on normalized display values
    # should clamp after exposure/brightness normalization; raw electron-domain
    # inputs can be far above 1.0 and would otherwise saturate to a gray image.
    return torch.stack([red, green, blue], dim=0)
