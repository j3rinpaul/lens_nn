import torch
import torchvision.models as models
import torchvision.transforms as transforms


import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

# Remove the final FC layer
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)

feature_extractor.eval()

# Freeze the network
for param in feature_extractor.parameters():
    param.requires_grad = False

def extract_features(image_tensor):
    """
    Extracts lighting-aware features from an AE image tensor (3, H, W).
    These features are chosen to survive auto-exposure tone-mapping
    and capture the lighting characteristics the camera struggled with.

    Returns a 1D tensor of shape (74,):
        - 30: RGB spatial histogram (2x2 grid, 10 bins, per channel)
        -  9: per-channel mean, std, clipping ratio
        -  9: spatial block brightness stats (2x2 grid mean + global contrast)
        - 12: highlight/shadow clipping per region
        - 14: color ratios and local variance features
    """
    r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    _, H, W = image_tensor.shape
    features = []

    # ── 1. RGB spatial histogram (30 dims) ────────────────────────────────────
    # Per-channel histogram in 2x2 blocks captures color + spatial lighting
    # Color info matters: warm lights → high R, cool → high B
    grid = 2
    h_step, w_step = H // grid, W // grid
    for row in range(grid):
        for col in range(grid):
            ys, ye = row * h_step, (row + 1) * h_step
            xs, xe = col * w_step, (col + 1) * w_step
            n_pixels = h_step * w_step
            for channel in [r, g, b]:
                block = channel[ys:ye, xs:xe]
                hist = torch.histc(block, bins=5, min=0.0, max=1.0)
                features.append(hist / n_pixels)
    # 2x2 x 3 channels x 5 bins = 60 dims  ← correct below
    # Actually: 4 blocks * 3 channels * 5 bins = 60 dims

    # ── 2. Per-channel global stats (9 dims) ──────────────────────────────────
    # Mean/std survive AE partially — AE targets overall brightness not per-channel
    # Clipping ratio: fraction of pixels that are saturated (>0.95) or crushed (<0.05)
    # High clipping → AE failed → strong signal for exposure correction
    for channel in [r, g, b]:
        ch_mean = channel.mean()
        ch_std  = channel.std()
        clipping = ((channel > 0.95) | (channel < 0.05)).float().mean()
        features.append(torch.stack([ch_mean, ch_std, clipping]))
    # 3 channels * 3 stats = 9 dims

    # ── 3. Local brightness contrast (5 dims) ─────────────────────────────────
    # High spatial variance in brightness = AE struggled with mixed lighting
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    block_means = []
    for row in range(grid):
        for col in range(grid):
            ys, ye = row * h_step, (row + 1) * h_step
            xs, xe = col * w_step, (col + 1) * w_step
            block_means.append(gray[ys:ye, xs:xe].mean())
    block_means_t = torch.stack(block_means)         # (4,)
    global_contrast = block_means_t.std()            # std across blocks = spatial unevenness
    features.append(block_means_t)
    features.append(global_contrast.unsqueeze(0))
    # 4 block means + 1 global contrast = 5 dims

    # ── Concatenate all ────────────────────────────────────────────────────────


    # with torch.no_grad():

    #     image_batch = image_tensor.unsqueeze(0).to(device)

    #     deep_features = feature_extractor(image_batch)

    #     deep_features = deep_features.flatten()

    handcrafted = torch.cat([f.flatten() for f in features])

    # final = torch.cat([handcrafted,deep_features.cpu()])

    # print(final.shape)

    # return final

    return handcrafted


# # Keep old function as alias so existing code doesn't break
# def extract_spatial_histogram(image_tensor, grid_size=2, bins=10):
#     r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     _, H, W = image_tensor.shape
#     h_step, w_step = H // grid_size, W // grid_size
#     block_histograms = []
#     n_pixels = h_step * w_step
#     for row in range(grid_size):
#         for col in range(grid_size):
#             ys, ye = row * h_step, (row + 1) * h_step
#             xs, xe = col * w_step, (col + 1) * w_step
#             block = gray[ys:ye, xs:xe]
#             hist = torch.histc(block, bins=bins, min=0.0, max=1.0)
#             block_histograms.append(hist / n_pixels)
#     return torch.cat(block_histograms)


if __name__ == '__main__':
    # Quick dim check
    dummy = torch.rand(3, 224, 224)
    out = extract_features(dummy)
    print(f'extract_features output dim: {out.shape[0]}')
    # 4 blocks * 3 channels * 5 bins = 60
    # 3 channels * 3 stats             =  9
    # 4 block means + 1 contrast        =  5
    # Total                             = 74
