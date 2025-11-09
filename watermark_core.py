"""
watermark_core.py
Updated Hybrid Classical + Learned Watermarking Core
----------------------------------------------------
Includes:
- Classical DWT + DCT + SVD embedding and semi-fragile verification (+ Reed–Solomon)
- Learned residual Encoder/Decoder (U-Net-like) with 112-bit payload
- Differentiable robustness attacks
- Fast training loop (AMP, TF32, channels_last, tuned DataLoader, optional RAM cache)
"""

# =========================
# Imports
# =========================
import os
import math
import random
import hashlib
import hmac
import warnings
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image, ImageOps

from skimage import color
from skimage.metrics import structural_similarity as ssim
import reedsolo

# =========================
# Device & performance flags
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
# Performance on Ampere (A100)
try:
    torch.set_float32_matmul_precision("high")  # enable TF32 matmul
except Exception:
    pass
cudnn.benchmark = True
cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# Basic image helpers
# =========================
def pil_to_cv2(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2):
    if img_cv2.ndim == 2:
        return Image.fromarray(img_cv2)
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def to_y_channel(rgb_img: np.ndarray):
    """Extract Y (luma) channel from RGB image (np.uint8 or [0,1] float)."""
    img = rgb_img.copy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = ycbcr[:, :, 0].astype(np.float32) / 255.0
    return Y

# =========================
# VGG feature extractor (for fragile digest)
# =========================
try:
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
except Exception:
    vgg = models.vgg16(pretrained=True).features.to(device).eval()

_VGG_LAYER_IDX = 10

class VGGHook:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activation = None
        self.handle = self.model[self.layer_idx].register_forward_hook(self.hook_fn)
    def hook_fn(self, module, _in, out):
        self.activation = out.detach()
    def close(self):
        self.handle.remove()

_vgg_hook = VGGHook(vgg, _VGG_LAYER_IDX)

_transform_vgg = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_vgg_descriptor(rgb_img: np.ndarray):
    """Return pooled VGG feature for fragile digest computation."""
    img_t = _transform_vgg(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = vgg(img_t)
        feat = _vgg_hook.activation
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze().cpu().numpy()
    return pooled.ravel()

# =========================
# HMAC (fragile digest)
# =========================
def compute_hmac_digest(feature_vec, key=b"secret_key", digest_bits=128):
    feat_bytes = feature_vec.astype(np.float32).tobytes()
    mac = hmac.new(key, feat_bytes, hashlib.sha256).digest()
    bitstr = "".join(f"{b:08b}" for b in mac)
    return bitstr[:digest_bits]

# =========================
# DWT + DCT + SVD helpers
# =========================
def dwt2_channel(channel):
    return pywt.dwt2(channel, "haar")

def idwt2_channel(coeffs):
    return pywt.idwt2(coeffs, "haar")

def block_dct(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def block_idct(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")

def svd_modify_and_reconstruct(A, Wbits, alpha=0.01):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_mod = S.copy()
    k = min(len(Wbits), len(S_mod))
    for i in range(k):
        bit = int(Wbits[i])
        S_mod[i] = S_mod[i] + alpha * (1 if bit == 1 else -1) * (np.mean(S) + 1e-8)
    A_mod = U @ np.diag(S_mod) @ Vt
    return A_mod

# =========================
# SIFT helpers (classical pipeline)
# =========================
def get_sift_keypoints(rgb_img, max_kp=64):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kps = sift.detect(gray, None)
    kps = sorted(kps, key=lambda x: -x.response)[:max_kp]
    return kps

def extract_patch(img_bgr, kp, patch_size=64):
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    half = patch_size // 2
    h, w = img_bgr.shape[:2]

    x1 = max(0, x - half)
    x2 = min(w, x + half)
    y1 = max(0, y - half)
    y2 = min(h, y + half)

    patch = img_bgr[y1:y2, x1:x2]
    valid_h, valid_w = patch.shape[:2]

    if valid_h == 0 or valid_w == 0:
        padded_dims = (patch_size, patch_size, img_bgr.shape[2]) if img_bgr.ndim == 3 else (patch_size, patch_size)
        padded = np.zeros(padded_dims, dtype=img_bgr.dtype)
        return padded, (x1, y1, x1, y1), (0, 0)

    if valid_h != patch_size or valid_w != patch_size:
        pad_bottom = patch_size - valid_h
        pad_right = patch_size - valid_w
        patch = cv2.copyMakeBorder(
            patch, top=0, bottom=pad_bottom, left=0, right=pad_right, borderType=cv2.BORDER_REPLICATE
        )

    bbox = (x1, y1, x1 + valid_w, y1 + valid_h)
    return patch, bbox, (valid_h, valid_w)

# =========================
# Classical embedding / extraction with RS
# =========================
def embed_digest_in_image(rgb_img, digest_bits, N_patches=8, patch_size=64, alpha=0.02):
    kp_list = get_sift_keypoints(rgb_img, max_kp=N_patches * 4)
    chosen = kp_list[:N_patches]
    if len(chosen) < N_patches:
        warnings.warn(f"Found only {len(chosen)} keypoints, requested {N_patches}. Using available keypoints.")
    if len(chosen) == 0:
        raise ValueError("No SIFT keypoints available for embedding.")

    Y = to_y_channel(rgb_img)
    Y_out = Y.copy()
    embed_locations = []
    total_bits = max(1, len(digest_bits))
    actual_patches = len(chosen)

    # rough capacity heuristic with an upper bound
    max_bits_per_patch = max(1, (patch_size // 2) * (patch_size // 2) // 2)
    max_bits_per_patch = min(max_bits_per_patch, 32)

    capacity = actual_patches * max_bits_per_patch
    if capacity < total_bits:
        raise ValueError(
            f"Insufficient embedding capacity: need {total_bits} bits but only {capacity} bits available "
            f"({actual_patches} patches * {max_bits_per_patch} bits/patch)."
        )

    bits_per_patch = max(1, math.ceil(total_bits / actual_patches))
    bits_per_patch = min(bits_per_patch, max_bits_per_patch)
    padded_bits = digest_bits.ljust(bits_per_patch * actual_patches, "0")

    for i, kp in enumerate(chosen):
        patch_bgr, (x1, y1, x2, y2), (valid_h, valid_w) = extract_patch(
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), kp, patch_size
        )
        if valid_h == 0 or valid_w == 0:
            continue
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        Yp = to_y_channel(patch_rgb)
        LL, (LH, HL, HH) = dwt2_channel(Yp)
        dct_LL = block_dct(LL)
        start = i * bits_per_patch
        bits = padded_bits[start : start + bits_per_patch]

        dct_mod = svd_modify_and_reconstruct(dct_LL, bits, alpha=alpha)
        LL_mod = block_idct(dct_mod)
        Yp_mod = idwt2_channel((LL_mod, (LH, HL, HH)))

        hpatch, wpatch = Yp_mod.shape
        target_h = min(valid_h, hpatch)
        target_w = min(valid_w, wpatch)
        if target_h <= 0 or target_w <= 0:
            continue
        Y_out[y1 : y1 + target_h, x1 : x1 + target_w] = Yp_mod[:target_h, :target_w]
        embed_locations.append((x1, y1, x1 + target_w, y1 + target_h))

    img = (rgb_img.copy()).astype(np.uint8)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycbcr[:, :, 0] = np.clip(Y_out * 255.0, 0, 255)
    out_rgb = cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return out_rgb, embed_locations, bits_per_patch

def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def extract_digest_with_confidence(rgb_img, N_patches=8, patch_size=64, bits_per_patch=16):
    kp_list = get_sift_keypoints(rgb_img, max_kp=N_patches * 4)
    chosen = kp_list[:N_patches]
    extracted_bits, confidences, boxes = [], [], []
    for kp in chosen:
        patch_bgr, (x1, y1, x2, y2), (valid_h, valid_w) = extract_patch(
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), kp, patch_size
        )
        if valid_h == 0 or valid_w == 0:
            continue
        Yp = to_y_channel(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
        LL, (LH, HL, HH) = dwt2_channel(Yp)
        dct_LL = block_dct(LL)
        U, S, Vt = np.linalg.svd(dct_LL, full_matrices=False)
        med = np.median(S)
        k = min(bits_per_patch, len(S))
        if k == 0:
            continue
        patch_bits = "".join("1" if (s - med) > 0 else "0" for s in S[:k])
        # confidence: separation of top singulars vs. median
        top_mean = float(np.mean(np.abs(S[:k] - med)))
        denom = float(np.mean(np.abs(S)) + 1e-9)
        conf = np.tanh(top_mean / denom)
        extracted_bits.append(patch_bits)
        confidences.append(conf)
        boxes.append((x1, y1, x2, y2))
    return extracted_bits, confidences, boxes

def weighted_majority_aggregate(extracted_bits_list, confidences, expected_len=None):
    if len(extracted_bits_list) == 0:
        return "", 0.0
    if expected_len is None:
        expected_len = max(len(s) for s in extracted_bits_list) if extracted_bits_list else 0
    if expected_len == 0:
        return "", 0.0

    votes = np.zeros((expected_len, 2), dtype=float)
    for s, conf in zip(extracted_bits_list, confidences):
        s_p = s.ljust(expected_len, "0")[:expected_len]
        for i, ch in enumerate(s_p):
            votes[i, int(ch)] += conf
    final_bits = "".join("1" if votes[i, 1] >= votes[i, 0] else "0" for i in range(expected_len))
    avg_conf = float(np.mean(confidences))
    return final_bits, avg_conf

def bits_to_bytes(bitstr: str) -> bytes:
    pad = (-len(bitstr)) % 8
    bitstr_padded = bitstr + ("0" * pad)
    b = bytes(int(bitstr_padded[i : i + 8], 2) for i in range(0, len(bitstr_padded), 8))
    return b

def bytes_to_bits(b: bytes, nbits: int = None) -> str:
    s = "".join(f"{byte:08b}" for byte in b)
    if nbits is None:
        return s
    return s[:nbits]

def rs_encode_bits(bitstr: str, nsym: int = 32) -> str:
    b = bits_to_bytes(bitstr)
    rsc = reedsolo.RSCodec(nsym)
    enc = rsc.encode(b)
    return bytes_to_bits(enc)

def rs_decode_bits(encoded_bitstr: str, orig_nbits: int, nsym: int = 32):
    try:
        enc_bytes = bits_to_bytes(encoded_bitstr)
        rsc = reedsolo.RSCodec(nsym)
        dec, _, _ = rsc.decode(enc_bytes)
        dec_bits = bytes_to_bits(dec, orig_nbits)
        return dec_bits, True, None
    except Exception as e:
        return None, False, str(e)

def embed_digest_with_rs(rgb_img, digest_bits, nsym=32, patch_size=64, min_patches=8, **kwargs):
    current_nsym = nsym
    last_error = None
    while current_nsym >= 4:  # Min 4 parity bytes
        enc_bits = rs_encode_bits(digest_bits, nsym=current_nsym)

        max_bits_per_patch = max(1, (patch_size // 2) * (patch_size // 2) // 2)
        max_bits_per_patch = min(max_bits_per_patch, 32)

        required_patches = math.ceil(len(enc_bits) / max_bits_per_patch)
        desired_patches = max(min_patches, required_patches)

        kwargs_local = dict(kwargs)
        N_patches = max(kwargs_local.pop("N_patches", desired_patches), desired_patches)

        try:
            wm_img, locations, bits_per_patch = embed_digest_in_image(
                rgb_img, enc_bits, N_patches=N_patches, patch_size=patch_size, **kwargs_local
            )
            return wm_img, locations, bits_per_patch, enc_bits, current_nsym
        except ValueError as e:
            last_error = e
            if current_nsym <= 4:
                break
            reduced_nsym = max(4, current_nsym // 2)
            warnings.warn(
                f"Embedding capacity limited; reducing RS parity bytes from {current_nsym} to {reduced_nsym}. Error: {e}",
                RuntimeWarning,
            )
            current_nsym = reduced_nsym
    raise ValueError(f"Failed to embed ECC-protected watermark: {last_error}")

def extract_and_decode_rs(rgb_img, orig_nbits=128, N_patches=8, bits_per_patch=16, nsym=32, encoded_bits_len=None):
    extracted_list, confs, boxes = extract_digest_with_confidence(
        rgb_img, N_patches=N_patches, patch_size=64, bits_per_patch=bits_per_patch
    )
    if encoded_bits_len is None:
        orig_bytes = math.ceil(orig_nbits / 8)
        expected_len = (orig_bytes + nsym) * 8
    else:
        expected_len = encoded_bits_len
    agg_bits, avg_conf = weighted_majority_aggregate(extracted_list, confs, expected_len=expected_len)
    trimmed_bits = agg_bits[:expected_len]
    decoded, ok, info = rs_decode_bits(trimmed_bits, orig_nbits, nsym=nsym)
    return decoded, ok, info, avg_conf, boxes, confs

def verify_semi_fragile(
    received_rgb,
    original_rgb,
    key=b"secret_key",
    N_patches=8,
    digest_bits=128,
    bits_per_patch=16,
    nsym=32,
    encoded_bits_len=None,
    T_accept=12,
    T_reject=30,
):
    decoded, ok, info, avg_conf, boxes, confs = extract_and_decode_rs(
        received_rgb,
        orig_nbits=digest_bits,
        N_patches=N_patches,
        bits_per_patch=bits_per_patch,
        nsym=nsym,
        encoded_bits_len=encoded_bits_len,
    )

    orig_feat = extract_vgg_descriptor(original_rgb)
    orig_digest = compute_hmac_digest(orig_feat, key=key, digest_bits=digest_bits)

    if ok and decoded == orig_digest:
        fr_result = "PASS"
    elif ok and decoded is not None:
        ham = hamming_distance(decoded, orig_digest)
        if ham <= T_accept:
            fr_result = "PASS"
        elif ham > T_reject:
            fr_result = "TAMPER"
        else:
            fr_result = "UNCERTAIN"
    else:
        fr_result = "TAMPER"
        avg_conf = (1.0 - avg_conf)

    return fr_result, avg_conf, boxes, confs

# =========================
# Classical attack simulators (optional utilities)
# =========================
def attack_jpeg(img, quality=75):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

def attack_resize(img, scale=0.8):
    h, w = img.shape[:2]
    newh, neww = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back

def attack_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def attack_noise(img, sigma=5.0):
    noise = np.random.normal(0, sigma, img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype("uint8")
    return out

def attack_rotate(img, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def attack_crop(img, crop_ratio=0.85):
    h, w = img.shape[:2]
    ch = max(1, int(h * crop_ratio))
    cw = max(1, int(w * crop_ratio))
    y1 = max(0, (h - ch) // 2)
    x1 = max(0, (w - cw) // 2)
    cropped = img[y1 : y1 + ch, x1 : x1 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def attack_patch_replace(img, size=0.15):
    h, w = img.shape[:2]
    ph, pw = int(h * size), int(w * size)
    x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
    x2, y2 = random.randint(0, w - pw), random.randint(0, h - ph)
    patch = img[y2 : y2 + ph, x2 : x2 + pw].copy()
    out = img.copy()
    out[y1 : y1 + ph, x1 : x1 + pw] = patch
    return out

# =========================
# Learned Encoder/Decoder (112 bits)
# =========================
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden=128, payload_len=112):
        super().__init__()
        self.hidden = hidden
        self.payload_len = payload_len
        self.payload_embed = nn.Sequential(
            nn.Linear(payload_len, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels + hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, dilation=2),
            nn.BatchNorm2d(hidden * 2),
            nn.SiLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2),
            nn.SiLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU()
        )
        self.out_conv = nn.Conv2d(hidden, in_channels, 1)

    def forward(self, x, payload_bits):
        if payload_bits is None:
            payload_bits = torch.zeros(x.size(0), self.payload_len, device=x.device, dtype=x.dtype)
        payload_feat = self.payload_embed(payload_bits)
        payload_feat = payload_feat.view(payload_feat.size(0), self.hidden, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x_in = torch.cat([x, payload_feat], dim=1)
        d1 = self.down1(x_in)
        p = self.pool(d1)
        d2 = self.down2(p)
        u = self.up1(d2)
        if u.shape[-2:] != d1.shape[-2:]:
            u = F.interpolate(u, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([u, d1], dim=1)
        r = self.fuse(fused)
        res = torch.tanh(self.out_conv(r)) * 0.2
        return res

class Decoder(nn.Module):
    def __init__(self, in_channels=3, payload_len=112, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2 * 8 * 8, 1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, payload_len)
        )
    def forward(self, x):
        f = self.conv(x)
        f = self.pool(f)
        out = self.fc(f)
        return out  # raw logits (use BCEWithLogitsLoss)

# =========================
# Differentiable Attack (for training)
# =========================
class DifferentiableAttack(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, imgs, severity: float = 1.0):
        x = imgs
        severity = float(max(0.0, min(0.6, severity)))
        b, c, h, w = x.shape

        # random affine jitter (rotation / translation / scale / shear)
        if random.random() < 0.4 + 0.4 * severity:
            angle = random.uniform(-10, 10) * severity
            translate = (
                int(random.uniform(-0.08, 0.08) * severity * w),
                int(random.uniform(-0.08, 0.08) * severity * h),
            )
            scale = random.uniform(0.75, 1.05)
            shear_x = random.uniform(-8, 8) * severity
            shear_y = random.uniform(-8, 8) * severity
            x = TF.affine(
                x,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(shear_x, shear_y),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.5,
            )

        # multi-scale blur / down-up sampling
        if random.random() < 0.5 + 0.3 * severity:
            scale = random.uniform(max(0.55, 0.85 - 0.3 * severity), 1.0)
            nh, nw = max(4, int(h * scale)), max(4, int(w * scale))
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        if random.random() < 0.25 + 0.35 * severity:
            sigma_min = 0.15
            sigma = random.uniform(sigma_min, 0.9 + 0.8 * severity)
            x = TF.gaussian_blur(x, kernel_size=3, sigma=sigma)

        if random.random() < 0.3 + 0.3 * severity:
            contrast = random.uniform(0.75, 1.25)
            x = torch.clamp(TF.adjust_contrast(x, contrast), 0.0, 1.0)

        if random.random() < 0.3 + 0.3 * severity:
            gamma = random.uniform(0.85, 1.2)
            x = torch.clamp(TF.adjust_gamma(x, gamma), 0.0, 1.0)

        if random.random() < 0.25 + 0.4 * severity:
            saturation = random.uniform(0.6, 1.4)
            x = torch.clamp(TF.adjust_saturation(x, saturation), 0.0, 1.0)

        if random.random() < 0.2 + 0.3 * severity:
            hue = random.uniform(-0.04, 0.04)
            x = torch.clamp(TF.adjust_hue(x, hue), 0.0, 1.0)

        # channel drop / mix
        if random.random() < 0.2 * severity:
            mask = torch.ones_like(x)
            ch = random.randrange(c)
            mask[:, ch, :, :] = mask[:, ch, :, :].mul(0.0)
            x = x * mask

        # light gaussian noise
        if random.random() < 0.6 + 0.3 * severity:
            noise_level = 0.008 + 0.025 * severity
            x = torch.clamp(x + torch.randn_like(x) * noise_level, 0, 1)

        return x

# =========================
# SimpleImageFolder dataset (fast, optional RAM cache)
# =========================
class SimpleImageFolder(Dataset):
    """
    - Recursively loads images under root_dir
    - Fixes EXIF orientation, forces RGB
    - Center-crops to square, resizes to image_size
    - Outputs CHW float32 tensors in [0,1]
    - cache_to_ram=True: preload all processed tensors for faster epochs
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: Union[str, Path], image_size: int = 256, cache_to_ram: bool = False, augment: bool = True):
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Image root not found: {self.root_dir}")
        self.paths: List[Path] = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        ]
        if not self.paths:
            raise RuntimeError(f"No images with extensions {sorted(self.IMG_EXTS)} under {self.root_dir}")

        self.image_size = int(image_size)
        self.cache_to_ram = bool(cache_to_ram)
        self.augment = bool(augment)
        self._cache: List[torch.Tensor] = []
        if self.cache_to_ram:
            print("⚡ Pre-loading images into RAM …")
            for p in self.paths:
                self._cache.append(self._load_to_tensor(p))

    def __len__(self) -> int:
        return int(len(self.paths))

    def _load_image_rgb(self, p: Path) -> Image.Image:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _center_crop_square(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))

    def _load_to_tensor(self, p: Path) -> torch.Tensor:
        img = self._load_image_rgb(p)
        img = self._center_crop_square(img)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return ten

    def _apply_augmentations(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return tensor

        x = tensor
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < 0.1:
            x = torch.flip(x, dims=[1])

        if torch.rand(1).item() < 0.6:
            angle = float(torch.empty(1).uniform_(-8, 8))
            translate = (
                int(float(torch.empty(1).uniform_(-0.08, 0.08)) * x.shape[2]),
                int(float(torch.empty(1).uniform_(-0.08, 0.08)) * x.shape[1]),
            )
            scale = float(torch.empty(1).uniform_(0.85, 1.05))
            shear_x = float(torch.empty(1).uniform_(-6, 6))
            shear_y = float(torch.empty(1).uniform_(-6, 6))
            x = TF.affine(
                x,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(shear_x, shear_y),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.5,
            )

        if torch.rand(1).item() < 0.5:
            brightness = float(torch.empty(1).uniform_(0.85, 1.15))
            x = torch.clamp(TF.adjust_brightness(x, brightness), 0.0, 1.0)
        if torch.rand(1).item() < 0.5:
            contrast = float(torch.empty(1).uniform_(0.8, 1.2))
            x = torch.clamp(TF.adjust_contrast(x, contrast), 0.0, 1.0)
        if torch.rand(1).item() < 0.4:
            saturation = float(torch.empty(1).uniform_(0.75, 1.25))
            x = torch.clamp(TF.adjust_saturation(x, saturation), 0.0, 1.0)
        if torch.rand(1).item() < 0.3:
            hue = float(torch.empty(1).uniform_(-0.03, 0.03))
            x = torch.clamp(TF.adjust_hue(x, hue), 0.0, 1.0)

        if torch.rand(1).item() < 0.3:
            sigma = float(torch.empty(1).uniform_(0.2, 1.0))
            x = TF.gaussian_blur(x, kernel_size=3, sigma=sigma)

        if torch.rand(1).item() < 0.35:
            channel_scale = torch.empty(3, 1, 1).uniform_(0.85, 1.15)
            x = torch.clamp(x * channel_scale, 0.0, 1.0)

        if torch.rand(1).item() < 0.4:
            noise = torch.randn_like(x) * float(torch.empty(1).uniform_(0.005, 0.02))
            x = torch.clamp(x + noise, 0.0, 1.0)

        return x

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache_to_ram:
            img = self._cache[int(idx)]
            if self.augment:
                img = img.clone()
        else:
            img = self._load_to_tensor(self.paths[int(idx)])
        img = self._apply_augmentations(img)
        return img

# =========================
# Training loop (fast)
# =========================
def train_residual_encoder(
    root_images: str,
    epochs: int = 20,
    batch_size: int = 64,     # try 64+ on A100; reduce if OOM
    payload_len: int = 112,
    lr: float = 1e-4,
    save_path: str = "best_residual_hybrid.pt",
    curriculum: bool = True,
    cache_to_ram: bool = False,
    max_attack_strength: float = 0.45,
    amp: bool = True,
    resume_from: str = None,
):
    """
    Fast training:
      - Mixed precision (AMP) + GradScaler
      - channels_last memory format
      - TF32 enabled on Ampere (A100)
      - Tuned DataLoader (num_workers, pin_memory, persistent_workers, prefetch_factor)
      - Optional RAM cache for the dataset
      - BCEWithLogitsLoss + residual L2 regularization
      - Adjustable attack curriculum (max_attack_strength) with stability guards
      - Optional AMP mixed precision (disable via amp=False)
      - Resume support via saved checkpoint (resume_from/save_path)
    """
    # ---- Data
    ds = SimpleImageFolder(root_images, image_size=256, cache_to_ram=cache_to_ram, augment=True)
    num_workers = max(2, (os.cpu_count() or 2) // 2)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    # ---- Models
    enc = Encoder(payload_len=payload_len).to(device).to(memory_format=torch.channels_last)
    dec = Decoder(payload_len=payload_len).to(device).to(memory_format=torch.channels_last)
    attack = DifferentiableAttack().to(device)

    # ---- Optim / Scheduler / AMP (new API + stronger LR, no weight decay)
    amp_enabled = bool(amp and device == "cuda")

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=lr,
        weight_decay=0.0      # explicitly no WD
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    best_ema = 0.0
    ema_acc = 0.0
    alpha = 0.2
    residual_ema = 0.0
    attack_limit = float(max(0.25, min(0.9, max_attack_strength)))
    attack_limit_init = attack_limit

    start_epoch = 0
    resume_path = None
    auto_resume = resume_from is None
    if resume_from:
        rp = Path(resume_from)
        if rp.exists():
            resume_path = rp
        else:
            print(f"⚠️ Resume path not found: {resume_from}")
    if auto_resume and resume_path is None:
        sp = Path(save_path)
        if sp.exists():
            resume_path = sp
            print(f"ℹ️ Auto-resuming from existing checkpoint at {save_path}")

    if resume_path and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device)
            enc.load_state_dict(ckpt["enc"])
            dec.load_state_dict(ckpt["dec"])
            best_ema = float(ckpt.get("best_ema", best_ema))
            ema_acc = float(ckpt.get("ema_acc", ema_acc))
            residual_ema = float(ckpt.get("residual_ema", residual_ema))
            attack_limit = float(min(max_attack_strength, ckpt.get("attack_limit", attack_limit)))
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"🔁 Resumed from {resume_path} @ epoch {start_epoch} | best_ema={best_ema*100:.2f}% atk_cap={attack_limit:.2f}")
        except Exception as e:
            print(f"⚠️ Failed to resume from {resume_path}: {e}")
            start_epoch = 0
            attack_limit = attack_limit_init

    if start_epoch >= epochs:
        print(f"✅ Already trained for {start_epoch} epochs (>= {epochs}). Nothing to do.")
        return

    for _ in range(start_epoch):
        sched.step()

    # Warm-up: no attacks for longer to build strong base
    warmup_epochs = max(8, epochs // 3)      # ~40% of training without attacks
    # Residual penalty ramp: start very light, grow slowly
    def residual_weight(e):
        return 0.003 + 0.012 * min(1.0, e / max(1, epochs - 1))

    last_epoch_ema = ema_acc if start_epoch > 0 else None
    for epoch in range(start_epoch + 1, epochs + 1):
        enc.train(); dec.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}")

        curriculum_progress = 0.0
        if curriculum and epoch > warmup_epochs:
            # Much slower attack ramping: square root makes it grow slower initially
            raw_progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            curriculum_progress = math.sqrt(raw_progress)  # slower initial growth
        
        # Strong damping if accuracy drops
        severity_dampen = 1.0
        if last_epoch_ema is not None:
            acc_drop = last_epoch_ema - ema_acc
            if acc_drop > 0.08:  # significant drop
                severity_dampen = 0.5
            elif acc_drop > 0.05:  # moderate drop
                severity_dampen = 0.7
            elif acc_drop > 0.02:  # small drop
                severity_dampen = 0.85
        
        adjusted_progress = curriculum_progress * severity_dampen

        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            B = imgs.size(0)
            target_bits = torch.randint(0, 2, (B, payload_len), device=device, dtype=torch.float32)

            attack_strength = 0.0
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # Encode residual & compose watermarked
                residual = enc(imgs, target_bits)
                watermarked = imgs + residual                      # keep linear path for grads
                watermarked_clamped = torch.clamp(watermarked, 0, 1)

                # Clean-path decode (aux loss)
                logits_clean = dec(watermarked_clamped)
                bce_clean = F.binary_cross_entropy_with_logits(logits_clean, target_bits)

                # Attacked path
                if epoch <= warmup_epochs or not curriculum:
                    attacked = watermarked_clamped
                    # During warmup, focus heavily on clean accuracy
                    attack_weight = 0.5  # reduce importance of attacked path during warmup
                else:
                    # Start with very weak attacks
                    attack_strength = min(adjusted_progress * 0.5, attack_limit)  # cap at 50% of progress
                    attack_weight = 0.8 + 0.2 * min(1.0, attack_strength / 0.2)  # gradually increase attacked weight
                    
                    # Apply attacks with probability based on strength
                    if random.random() < 0.5 + 0.3 * attack_strength:
                        attacked = attack(watermarked_clamped, severity=attack_strength)
                    else:
                        attacked = watermarked_clamped
                    
                    # Add light noise more conservatively
                    if random.random() < 0.4 + 0.2 * attack_strength:
                        noise_mag = 0.004 + 0.015 * attack_strength
                        attacked = torch.clamp(attacked + torch.randn_like(attacked) * noise_mag, 0, 1)
                    
                    # Only add blur at higher attack strengths
                    if attack_strength > 0.25 and random.random() < 0.1 + 0.25 * max(0.0, attack_strength - 0.25):
                        sigma = random.uniform(0.25, 0.8) * (1.0 + 0.3 * attack_strength)
                        attacked = TF.gaussian_blur(attacked, kernel_size=3, sigma=sigma)

                logits_att = dec(attacked)
                bce_att = F.binary_cross_entropy_with_logits(logits_att, target_bits)

                # Residual regularization (ramped)
                res_l2 = (residual ** 2).mean()
                lam = residual_weight(epoch - 1)   # epoch-1 so epoch1 uses smallest weight

                # Total loss: balance between clean and attacked, with residual penalty
                # During early training, prioritize clean accuracy more
                clean_weight = 1.0 if epoch <= warmup_epochs else 0.6
                loss = attack_weight * bce_att + clean_weight * bce_clean + lam * res_l2

            if not torch.isfinite(loss):
                print(f"⚠️ Non-finite loss encountered at epoch {epoch} (attack_cap={attack_limit:.2f}). Reducing attack intensity.")
                attack_limit = max(0.20, attack_limit * 0.6)
                opt.zero_grad(set_to_none=True)
                # Skip scaler.update() - it requires a prior step() call
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                residual_abs = residual.detach().abs().mean().item()
                residual_ema = 0.1 * residual_abs + 0.9 * residual_ema
                probs = torch.sigmoid(logits_att)
                preds = (probs > 0.5).float()
                bit_acc = preds.eq(target_bits).float().mean().item()
                ema_acc = alpha * bit_acc + (1 - alpha) * ema_acc

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bitAcc=f"{bit_acc*100:.1f}%",
                EMA=f"{ema_acc*100:.1f}%",
                lam=f"{lam:.3f}",
                res=f"{residual_ema:.4f}",
                atk=f"{attack_strength:.2f}",
                atkCap=f"{attack_limit:.2f}"
            )

        # Dynamic attack cap adjustment - more conservative
        if ema_acc < 0.60 and attack_limit > 0.25:
            attack_limit = max(0.25, attack_limit * 0.85)
            print(f"📉 Accuracy low ({ema_acc*100:.1f}%), reducing attack_cap to {attack_limit:.2f}")
        
        if residual_ema > 0.13 and attack_limit > 0.25:
            attack_limit = max(0.25, attack_limit * 0.90)
            print(f"📊 High residual ({residual_ema:.4f}), reducing attack_cap to {attack_limit:.2f}")
        
        # Only increase attack if both accuracy AND residual are good
        if last_epoch_ema is not None and ema_acc > max(0.65, last_epoch_ema + 0.02) and residual_ema < 0.10:
            attack_limit = min(max_attack_strength, attack_limit + 0.015)
            print(f"📈 Good progress, increasing attack_cap to {attack_limit:.2f}")
        
        attack_limit = float(max(0.20, min(max_attack_strength, attack_limit)))

        sched.step()
        if ema_acc > best_ema:
            best_ema = ema_acc
            torch.save({
                "enc": enc.state_dict(),
                "dec": dec.state_dict(),
                "payload_len": payload_len,
                "best_ema": best_ema,
                "ema_acc": ema_acc,
                "attack_limit": attack_limit,
                "residual_ema": residual_ema,
                "epoch": epoch,
            }, save_path)
        last_epoch_ema = ema_acc

    torch.save({
        "enc": enc.state_dict(),
        "dec": dec.state_dict(),
        "payload_len": payload_len,
        "best_ema": best_ema,
        "ema_acc": ema_acc,
        "attack_limit": attack_limit,
        "residual_ema": residual_ema,
        "epoch": epochs,
    }, save_path)
    print(f"✅ Done. Best EMA BitAcc: {best_ema*100:.2f}% | Saved: {save_path}")

# =========================
# Hybrid verification (fragile + robust fusion)
# =========================
def robust_verify_placeholder(received_img):
    # Placeholder if learned decoder not provided.
    return True, 0.85

def fuse_decisions(fr_result, fr_conf, robust_ok, robust_conf):
    if fr_result == "TAMPER":
        if robust_ok and robust_conf > 0.9 and fr_conf < 0.2:
            return "DISPUTED"
        return "TAMPER"
    if fr_result == "PASS":
        return "PASS" if robust_ok else "PASS_NO_OWNERSHIP"
    if fr_result == "UNCERTAIN":
        if robust_ok and robust_conf > 0.7:
            return "POSSIBLE_PASS"
        if fr_conf < 0.4 and not robust_ok:
            return "FLAG_FOR_REVIEW"
        return "UNCERTAIN"

def combined_verification_pipeline(
    received_rgb,
    original_rgb,
    key=b"my_secret_key",
    orig_nbits=128,
    nsym=32,
    N_patches=8,
    bits_per_patch=16,
    encoded_bits_len=None,
    learned_decoder=None,
    payload_tensor=None,
):
    # 1) classical semi-fragile
    fr_result, fr_conf, patch_boxes, patch_confidences = verify_semi_fragile(
        received_rgb,
        original_rgb,
        key=key,
        digest_bits=orig_nbits,
        N_patches=N_patches,
        bits_per_patch=bits_per_patch,
        nsym=nsym,
        encoded_bits_len=encoded_bits_len,
    )

    # 2) learned robust
    if learned_decoder is not None and payload_tensor is not None:
        try:
            img_t = torch.from_numpy(received_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            img_t = F.interpolate(img_t, size=(256, 256), mode="bilinear", align_corners=False)
            with torch.no_grad():
                logits = learned_decoder(img_t)
                preds = (torch.sigmoid(logits) > 0.5).float()
                ber = (preds != payload_tensor).float().mean().item()
            robust_ok = ber < 0.01
            robust_conf = 1.0 - (ber * 2)
        except Exception as e:
            print(f"[WARN] Learned decoder failed: {e}")
            robust_ok, robust_conf, ber = False, 0.0, 1.0
    else:
        robust_ok, robust_conf = robust_verify_placeholder(received_rgb)
        ber = 1.0 - robust_conf

    fused = fuse_decisions(fr_result, fr_conf, robust_ok, robust_conf)
    return {
        "fragile_result": fr_result,
        "fragile_conf": float(fr_conf),
        "robust_ok": bool(robust_ok),
        "robust_conf": float(robust_conf),
        "fused_decision": fused,
        "payload_ber": float(ber),
        "patch_boxes": patch_boxes,
        "patch_confidences": patch_confidences,
    }

# =========================
# Tamper localization (visuals)
# =========================
def compute_structural_heatmap(original_rgb, suspect_rgb, window_size=21, sigma=1.5):
    orig_gray = color.rgb2gray(original_rgb)
    suspect_gray = color.rgb2gray(suspect_rgb)
    score, diff = ssim(
        orig_gray,
        suspect_gray,
        win_size=window_size,
        gaussian_weights=True,
        sigma=sigma,
        use_sample_covariance=False,
        full=True,
        data_range=1.0,
    )
    heatmap = 1.0 - diff
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap, score

def build_patch_confidence_map(image_shape, patch_boxes, patch_confidences):
    h, w = image_shape[:2]
    accumulator = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2), conf in zip(patch_boxes, patch_confidences):
        accumulator[y1:y2, x1:x2] += conf
        counts[y1:y2, x1:x2] += 1.0
    mask = counts > 0
    accumulator[mask] /= counts[mask]
    return accumulator

def fuse_heatmaps(structural_heatmap, patch_heatmap, weight_structural=0.7):
    patch_rescaled = (patch_heatmap - patch_heatmap.min()) / (patch_heatmap.max() - patch_heatmap.min() + 1e-8)
    fused = np.clip(weight_structural * structural_heatmap + (1 - weight_structural) * patch_rescaled, 0.0, 1.0)
    return fused

def overlay_heatmap(rgb_img, heatmap, cmap="inferno", alpha=0.6):
    import matplotlib.cm as cm
    rgb_norm = rgb_img.astype(np.float32) / 255.0
    colormap = cm.get_cmap(cmap)(heatmap)[..., :3]
    overlay = np.clip(alpha * colormap + (1 - alpha) * rgb_norm, 0.0, 1.0)
    return overlay

def render_tamper_visualization(original_rgb, suspect_rgb, pipeline_result):
    structural_heatmap, ssim_score = compute_structural_heatmap(original_rgb, suspect_rgb)
    patch_boxes = pipeline_result.get("patch_boxes") or []
    patch_confidences = pipeline_result.get("patch_confidences") or []
    if len(patch_boxes) == 0 or len(patch_confidences) == 0:
        fused_heatmap = structural_heatmap
    else:
        patch_map = build_patch_confidence_map(original_rgb.shape, patch_boxes, patch_confidences)
        fused_heatmap = fuse_heatmaps(structural_heatmap, patch_map)
    overlay = overlay_heatmap(suspect_rgb, fused_heatmap)
    return {
        "structural_heatmap": structural_heatmap,
        "fused_heatmap": fused_heatmap,
        "overlay": overlay,
        "ssim_score": ssim_score,
    }
