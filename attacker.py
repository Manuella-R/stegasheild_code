"""
Attacker module with a diverse suite of deterministic, 
model-free (classical) image processing attacks.
"""

import os
from pathlib import Path
import json
import random
import numpy as np
from PIL import Image, ImageFilter
from utils import set_seed, ensure_dir, now_iso
import cv2

# --- Noise Attacks ---

def attack_noise(np_img: np.ndarray, sigma: float = 5.0):
    """Adds Gaussian noise."""
    img = np_img.astype(np.float32)
    noise = np.random.normal(0, sigma, img.shape)
    out = np.clip(img + noise, 0, 255).astype('uint8')
    return out

def attack_salt_pepper_noise(np_img: np.ndarray, salt_vs_pepper_ratio=0.5, amount=0.005):
    """Adds salt and pepper noise."""
    out = np_img.copy()
    num_salt = np.ceil(amount * np_img.size * salt_vs_pepper_ratio)
    num_pepper = np.ceil(amount * np_img.size * (1. - salt_vs_pepper_ratio))
    
    # Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_img.shape]
    out[coords[0], coords[1], coords[2]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_img.shape]
    out[coords[0], coords[1], coords[2]] = 0
    return out

# --- Blurring / Filtering Attacks ---

def attack_blur(np_img: np.ndarray, radius: float = 2.0):
    """Applies Gaussian blur. 'radius' is used to calculate ksize."""
    ksize = max(1, int(radius * 2) // 2 * 2 + 1) # Must be odd
    return cv2.GaussianBlur(np_img, (ksize,ksize), 0)

def attack_median_blur(np_img: np.ndarray, ksize: int = 3):
    """Applies Median blur."""
    ksize = max(1, int(ksize) // 2 * 2 + 1) # Must be odd
    return cv2.medianBlur(np_img, ksize)

def attack_average_blur(np_img: np.ndarray, ksize: int = 3):
    """Applies an averaging/box blur."""
    ksize = max(1, int(ksize))
    return cv2.blur(np_img, (ksize, ksize))

def attack_sharpen(np_img: np.ndarray, strength: float = 0.5):
    """Applies a sharpening kernel."""
    k = np.array([[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]], dtype=np.float32)
    identity = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.float32)
    kernel = identity * (1.0 - strength) + k * strength
    out = cv2.filter2D(np_img.astype(np.float32), -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

# --- Geometric Attacks ---

def attack_resize(np_img: np.ndarray, scale: float = 0.5):
    """Downscales and then upscales an image."""
    h,w = np_img.shape[:2]
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    im2 = cv2.resize(np_img, new_size, interpolation=cv2.INTER_AREA)
    im3 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_LINEAR)
    return im3

def attack_crop(np_img: np.ndarray, ratio: float = 0.8):
    """Randomly crops a portion and resizes it back to original."""
    h, w = np_img.shape[:2]
    nh, nw = int(h * ratio), int(w * ratio)
    if w <= nw or h <= nh: return np_img # Avoid crop if ratio > 1
    
    left = random.randint(0, max(0, w - nw))
    top = random.randint(0, max(0, h - nh))
    crop = np_img[top:top+nh, left:left+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def attack_rotate(np_img: np.ndarray, angle: float = 5.0):
    """Rotates an image and fills borders with reflection."""
    h,w = np_img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    return cv2.warpAffine(np_img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

def attack_affine_transform(np_img: np.ndarray, shear: float = 0.1, scale: float = 0.9):
    """Applies a combination of shear, scale, and minor translation."""
    h, w = np_img.shape[:2]
    M = np.array([[1, shear, 0],
                  [0, 1, 0]], dtype=np.float32)
    M[0,0] *= scale
    M[1,1] *= scale
    M[0,2] = random.uniform(-w*0.05, w*0.05) # Random X translation
    M[1,2] = random.uniform(-h*0.05, h*0.05) # Random Y translation
    
    return cv2.warpAffine(np_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def attack_perspective_transform(np_img: np.ndarray, magnitude: float = 0.1):
    """Applies a random perspective warp."""
    h, w = np_img.shape[:2]
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    
    mw, mh = w * magnitude, h * magnitude
    pts2 = np.float32([
        [random.uniform(-mw, mw), random.uniform(-mh, mh)],
        [w - random.uniform(-mw, mw), random.uniform(-mh, mh)],
        [random.uniform(-mw, mw), h - random.uniform(-mh, mh)],
        [w - random.uniform(-mw, mw), h - random.uniform(-mh, mh)]
    ])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(np_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# --- Photometric & Other Attacks ---

def attack_jpeg(np_img: np.ndarray, quality: int = 75):
    """Applies JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    is_success, encimg = cv2.imencode('.jpg', cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR), encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

def attack_brightness_contrast(np_img: np.ndarray, alpha: float = 1.3, beta: float = 10.0):
    """Adjusts brightness (beta) and contrast (alpha)."""
    return cv2.convertScaleAbs(np_img, alpha=alpha, beta=beta)

def attack_gamma_correction(np_img: np.ndarray, gamma: float = 0.8):
    """Applies gamma correction."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(np_img, table)

def attack_color_jitter(np_img: np.ndarray, hue: float = 0.05, saturation: float = 0.3):
    """Randomly shifts hue and scales saturation."""
    img_hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(img_hsv)
    
    hue_shift = random.uniform(-hue, hue) * 180
    h = (h + hue_shift) % 180
    
    sat_scale = 1.0 + random.uniform(-saturation, saturation)
    s *= sat_scale
    
    img_hsv = cv2.merge([h, s, v])
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

def attack_patch_replace(np_img: np.ndarray, size: float = 0.15):
    """Simulates self-splicing by copying one patch over another."""
    h, w = np_img.shape[:2]
    ph, pw = int(h * size), int(w * size)
    x1, y1 = random.randint(0, w - pw -1), random.randint(0, h - ph -1)
    x2, y2 = random.randint(0, w - pw -1), random.randint(0, h - ph -1)
    patch = np_img[y2:y2+ph, x2:x2+pw].copy()
    out = np_img.copy()
    out[y1:y1+ph, x1:x1+pw] = patch
    return out

def attack_text_overlay(np_img: np.ndarray, text="WATERMARK", size=1.0, opacity=0.5):
    """Adds semi-transparent text to the image."""
    out = np_img.copy()
    h, w = out.shape[:2]
    font_size = (w / 800) * size
    color = (
        random.randint(200, 255), 
        random.randint(200, 255), 
        random.randint(200, 255)
    )
    x = random.randint(0, max(1, int(w * 0.5)))
    y = random.randint(int(h * 0.75), max(1, h - 20))
    
    overlay = out.copy()
    cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, int(font_size * 2), cv2.LINE_AA)
    
    return cv2.addWeighted(overlay, opacity, out, 1 - opacity, 0)

def attack_channel_drop(np_img: np.ndarray, channel_idx=None):
    """Sets one color channel (R, G, or B) to zero."""
    if channel_idx is None:
        channel_idx = random.randint(0, 2)
    out = np_img.copy()
    out[:, :, channel_idx] = 0
    return out

# --- Main Attack Mapping ---

ATTACK_MAP = {
    # Noise
    'jpeg': attack_jpeg,
    'noise': attack_noise,
    'salt_pepper': attack_salt_pepper_noise,
    
    # Blurs/Filters
    'blur': attack_blur,
    'median_blur': attack_median_blur,
    'average_blur': attack_average_blur,
    'sharpen': attack_sharpen,
    
    # Geometric
    'resize': attack_resize,
    'crop': attack_crop,
    'rotate': attack_rotate,
    'affine': attack_affine_transform,
    'perspective': attack_perspective_transform,
    
    # Photometric & Other
    'brightness_contrast': attack_brightness_contrast,
    'gamma': attack_gamma_correction,
    'color_jitter': attack_color_jitter,
    'patch_replace': attack_patch_replace,
    'text_overlay': attack_text_overlay,
    'channel_drop': attack_channel_drop,
}

def apply_attacks(input_img_path: str, output_dir: str, attacks: list, seed: int = 0):
    """
    Applies a list of attack specifications to an image.
    
    Args:
        input_img_path: Path to the source image.
        output_dir: Folder to save attacked images.
        attacks: A list of attack dicts, e.g., [{'type': 'jpeg', 'quality': 70}, ...]
        seed: Random seed for reproducible attacks.
    """
    set_seed(seed)
    ensure_dir(output_dir)
    try:
        img = Image.open(input_img_path).convert('RGB')
        np_img = np.array(img)
    except Exception as e:
        print(f"Failed to load image {input_img_path}: {e}")
        return []

    rows = []
    base_name = Path(input_img_path).stem
    
    for i, a_spec in enumerate(attacks):
        typ = a_spec.get('type')
        fn = ATTACK_MAP.get(typ)
        if fn is None:
            print(f"Warning: Unknown attack type '{typ}', skipping.")
            continue
            
        params = a_spec.copy()
        params.pop('type', None)
        
        try:
            out_np = fn(np_img, **params)
            
            suffix = f"_{typ}"
            if params:
                kv = '_'.join([f"{k}{v}" for k, v in params.items()])
                suffix += f"_{kv}"
            
            out_name = f"{base_name}{suffix}.jpg" # Standardize output to JPG
            out_path = Path(output_dir) / out_name
            Image.fromarray(out_np.astype('uint8')).save(out_path, quality=95)
            
            rows.append({
                'input': input_img_path,
                'output': str(out_path),
                'attack_type': typ,
                'attack_params': json.dumps(params),
                'seed': seed,
                'timestamp': now_iso()
            })
        except Exception as e:
            print(f"Error applying attack {typ} to {input_img_path}: {e}")
            
    return rows