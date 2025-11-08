"""Verifier adapter. This file is now functional."""
import numpy as np
from PIL import Image
from utils import now_iso
import torch
import torch.nn.functional as F
from typing import Tuple, Dict

# Import all logic from the core file
import watermark_core as core

# Import the model registry from embedder
# This is safe as embedder.py no longer imports verifier.py
from embedder import REGISTRY, bytes_to_tensor

def external_extract(np_img: np.ndarray, np_original_img: np.ndarray, params: dict) -> Tuple[bytes, Dict]:
    """
    Implements the full HYBRID extraction/verification pipeline.
    1. Classical SIFT/DWT/SVD verification (for fragile digest)
    2. Learned Residual extraction (for robust payload)
    3. Fuses the results.
    """
    
    # --- Parameters ---
    payload_bytes = params.get('payload_bytes', b'StegaShield_v1')
    if not isinstance(payload_bytes, bytes):
        payload_bytes = str(payload_bytes).encode('utf-8')
        
    payload_len = REGISTRY.payload_len
    payload_t = bytes_to_tensor(payload_bytes, payload_len).unsqueeze(0)

    digest_bits = params.get('digest_bits', 128)
    hmac_key = params.get('hmac_key', b'StegaShield_key')
    nsym = params.get('nsym_used', 16) # Use the value from embedding if available
    n_patches = params.get('N_patches', 8)
    bits_per_patch = params.get('bits_per_patch', 16)
    encoded_len = params.get('encoded_len_bits', None)

    # --- Load Models ---
    dec = REGISTRY.get_decoder()

    # --- Run Combined Pipeline ---
    # This core function handles both classical and learned extraction
    try:
        verification_results = core.combined_verification_pipeline(
            np_img,
            np_original_img,
            key=hmac_key,
            orig_nbits=digest_bits,
            nsym=nsym,
            N_patches=n_patches,
            bits_per_patch=bits_per_patch,
            encoded_bits_len=encoded_len,
            learned_decoder=dec,
            payload_tensor=payload_t.squeeze(0) # Pass the target payload
        )
    except Exception as e:
        print(f"Error during verification pipeline: {e}")
        return b'', {'error': str(e), 'payload_ber': 1.0, 'fragile_conf': 0.0, 'robust_conf': 0.0, 'fused_decision': 'UNCERTAIN'}

    # Return the *known* payload and the results dict
    return payload_bytes, verification_results


def extract_and_verify(image_path: str, original_image_path: str, params: dict = None) -> dict:
    """
    Wrapper to load images and call external_extract.
    This version requires the original image for comparison.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        np_img = np.array(img)
    except Exception as e:
        return {'error': f"Failed to load suspect image: {e}", 'payload_ber': 1.0}

    try:
        orig_img = Image.open(original_image_path).convert('RGB')
        np_original_img = np.array(orig_img)
    except Exception as e:
        return {'error': f"Failed to load original image: {e}", 'payload_ber': 1.0}

    
    payload, dbg = external_extract(np_img, np_original_img, params or {})
    
    out = {
        'image_path': image_path,
        'original_image_path': original_image_path,
        'payload_ber': dbg.get('payload_ber', None),
        'robust_conf': dbg.get('robust_conf', None),
        'fragile_conf': dbg.get('fragile_conf', None),
        'fused_decision': dbg.get('fused_decision', 'UNCERTAIN'),
        'extracted_payload': payload.decode('utf-8', errors='ignore'),
        'timestamp': now_iso()
    }
    return out