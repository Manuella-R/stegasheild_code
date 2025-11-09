"""Verifier adapter. This file is now functional and aligned with 112-bit payloads."""
from typing import Tuple, Dict, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# Import all logic from the core file
import watermark_core as core
from utils import now_iso

# Import the model registry + helper from embedder (safe import)
from embedder import REGISTRY, bytes_to_tensor


def _to_tensor_256(np_img: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_t = torch.from_numpy(np_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img_t = F.interpolate(img_t, size=(256, 256), mode="bilinear", align_corners=False)
    return img_t


def external_extract(np_img: np.ndarray, np_original_img: np.ndarray, params: dict) -> Tuple[bytes, Dict]:
    """
    HYBRID extraction/verification pipeline:
    1) Classical SIFT/DWT/SVD + RS (semi-fragile digest check)
    2) Learned Decoder payload extraction (robust ownership)
    3) Fusion of decisions
    """
    # --- Resolve payload & params ---
    payload_bytes = params.get("payload_bytes", b"StegaShield_v1")
    if not isinstance(payload_bytes, (bytes, bytearray)):
        payload_bytes = str(payload_bytes).encode("utf-8")

    payload_len = REGISTRY.payload_len
    payload_t = bytes_to_tensor(payload_bytes, payload_len).unsqueeze(0)  # (1, payload_len)

    digest_bits = int(params.get("digest_bits", 128))
    hmac_key = params.get("hmac_key", b"StegaShield_key")
    nsym = int(params.get("nsym_used", params.get("nsym", 16)))  # Prefer value from embedding if present
    n_patches = int(params.get("N_patches", 8))
    bits_per_patch = int(params.get("bits_per_patch", 16))
    encoded_len = params.get("encoded_len_bits", None)

    # --- Load Decoder model ---
    dec = REGISTRY.get_decoder()

    # --- Run the combined verification pipeline ---
    try:
        verification_results = core.combined_verification_pipeline(
            np_img,
            np_original_img,
            key=hmac_key,
            orig_nbits=digest_bits,
            nsym=nsym,
            N_patches=n_patches,
            bits_per_patch=bits_per_patch,
            encoded_bits_len=encoded_len,      # NOTE: fixed name matches core.extract_and_decode_rs
            learned_decoder=dec,
            payload_tensor=payload_t.squeeze(0),
        )
    except Exception as e:
        print(f"[ERROR] Verification pipeline failed: {e}")
        return b"", {
            "error": str(e),
            "payload_ber": 1.0,
            "fragile_conf": 0.0,
            "robust_conf": 0.0,
            "fused_decision": "UNCERTAIN",
        }

    # Return the known payload and diagnostics
    return payload_bytes, verification_results


def extract_and_verify(image_path: str, original_image_path: str, params: dict = None) -> dict:
    """
    Convenience wrapper to load images and call external_extract.
    Requires the original (unwatermarked) image for the semi-fragile check.
    """
    params = params or {}

    try:
        img = Image.open(image_path).convert("RGB")
        np_img = np.array(img)
    except Exception as e:
        return {"error": f"Failed to load suspect image: {e}", "payload_ber": 1.0}

    try:
        orig_img = Image.open(original_image_path).convert("RGB")
        np_original_img = np.array(orig_img)
    except Exception as e:
        return {"error": f"Failed to load original image: {e}", "payload_ber": 1.0}

    payload, dbg = external_extract(np_img, np_original_img, params or {})

    out = {
        "image_path": image_path,
        "original_image_path": original_image_path,
        "payload_ber": dbg.get("payload_ber", None),
        "robust_conf": dbg.get("robust_conf", None),
        "fragile_conf": dbg.get("fragile_conf", None),
        "fused_decision": dbg.get("fused_decision", "UNCERTAIN"),
        "extracted_payload": payload.decode("utf-8", errors="ignore"),
        "timestamp": now_iso(),
    }
    # include classical stats if present
    err = dbg.get("error")
    if err:
        out["error"] = err
    return out
