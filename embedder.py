"""Embedder adapter. This file is now functional."""
import os
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
from utils import set_seed, ensure_dir, now_iso
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Import all logic from the core file
import watermark_core as core

# --- Global Model Registry ---
# This loads models once to avoid slow re-loading on every call
class ModelRegistry:
    def __init__(self, model_path='best_residual_hybrid.pt', payload_len=64):
        self.enc = None
        self.dec = None
        self.model_path = model_path
        self.payload_len = payload_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_encoder(self):
        if self.enc is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}. Please run hybrid_train.py first.")
            self.enc = core.Encoder(payload_len=self.payload_len).to(self.device)
            state = torch.load(self.model_path, map_location=self.device)
            self.enc.load_state_dict(state['enc'])
            self.enc.eval()
            print("Loaded learned Encoder model.")
        return self.enc

    def get_decoder(self):
        if self.dec is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}. Please run hybrid_train.py first.")
            self.dec = core.Decoder(payload_len=self.payload_len).to(self.device)
            state = torch.load(self.model_path, map_location=self.device)
            self.dec.load_state_dict(state['dec'])
            self.dec.eval()
            print("Loaded learned Decoder model.")
        return self.dec

# Instantiate the registry
# NOTE: payload_len must match the 'payload_bytes' in generate_dataset.py
REGISTRY = ModelRegistry(payload_len=112) # 14 bytes * 8 bits/byte = 112 bits

def bytes_to_tensor(payload_bytes: bytes, payload_len: int) -> torch.Tensor:
    """Converts bytes to a float tensor of bits."""
    bitstr = core.bytes_to_bits(payload_bytes, nbits=payload_len)
    bitstr = bitstr.ljust(payload_len, '0') # Pad if too short
    bits = [float(b) for b in bitstr]
    return torch.tensor(bits, dtype=torch.float32).to(REGISTRY.device)

def external_embed(np_img: np.ndarray, payload: bytes, params: dict) -> Tuple[np.ndarray, dict]:
    """
    Implements the full HYBRID embedding pipeline.
    1. Classical SIFT/DWT/SVD embed (for fragile digest)
    2. Learned Residual embed (for robust payload)
    """
    # --- 1. Classical Semi-Fragile Embed ---
    # This embed carries the HMAC digest for tamper localization
    digest_bits = params.get('digest_bits', 128)
    hmac_key = params.get('hmac_key', b'StegaShield_key')
    nsym = params.get('nsym', 16) # Reduced from 32 for better capacity
    
    feat = core.extract_vgg_descriptor(np_img)
    digest = core.compute_hmac_digest(feat, key=hmac_key, digest_bits=digest_bits)
    
    try:
        classical_wm_np, locations, bits_per_patch, enc_bits, nsym_used = core.embed_digest_with_rs(
            np_img,
            digest_bits=digest,
            nsym=nsym,
            patch_size=64,
            min_patches=8,
            alpha=0.02
        )
        meta = {
            'digest': digest,
            'encoded_len_bits': len(enc_bits),
            'bits_per_patch': bits_per_patch,
            'N_patches': len(locations),
            'nsym_used': nsym_used,
            'classical_embed_success': True
        }
    except Exception as e:
        print(f"Warning: Classical embed failed: {e}. Skipping classical part.")
        classical_wm_np = np_img
        meta = {'classical_embed_success': False, 'error': str(e)}

    # --- 2. Learned Robust Embed ---
    # This embed carries the actual payload (e.g., b'StegaShield_v1')
    enc = REGISTRY.get_encoder()
    payload_len = REGISTRY.payload_len
    
    # Convert payload bytes to bit tensor
    payload_t = bytes_to_tensor(payload, payload_len).unsqueeze(0)
    
    # Convert classically watermarked image to tensor
    img_t = torch.from_numpy(classical_wm_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(REGISTRY.device)
    img_t = F.interpolate(img_t, size=(256, 256), mode='bilinear', align_corners=False)

    with torch.no_grad():
        residual = enc(img_t, payload_t)
        
    # Resize residual to match original image size
    residual_fullsize = F.interpolate(residual, size=(np_img.shape[0], np_img.shape[1]), mode='bilinear', align_corners=False)
    
    # Add residual and convert back to numpy
    # NOTE: The original code added residual to img_t (resized 256x256)
    # This is likely a bug. It should be added to the full-size classical_wm_np
    
    # Convert classical_wm_np to tensor for addition
    classical_wm_t_full = torch.from_numpy(classical_wm_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(REGISTRY.device)

    final_wm_t = torch.clamp(classical_wm_t_full + residual_fullsize, 0.0, 1.0)
    final_wm_np = (final_wm_t.squeeze(0).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
    
    meta['robust_payload_len'] = payload_len
    
    return final_wm_np, meta

# --- This function is now provided by verifier.py ---
# We keep the import here for the embed_image check
from verifier import external_extract

def embed_image(input_path: str, output_path: str, payload: bytes = None, params: dict = None, seed: int = 0) -> Dict:
    set_seed(seed)
    ensure_dir(os.path.dirname(output_path) or '.')
    img = Image.open(input_path).convert('RGB')
    np_img = np.array(img)
    
    # Get payload from params or use default
    payload_bytes = payload or params.get('payload_bytes', b'DEFAULT_PAYLOAD')
    if not isinstance(payload_bytes, bytes):
        payload_bytes = str(payload_bytes).encode('utf-8')
        
    params_full = params or {}
    params_full['payload_bytes'] = payload_bytes # Ensure params has it for extractor
    
    wm_np, dbg = external_embed(np_img, payload_bytes, params_full)
    
    out_img = Image.fromarray(wm_np.astype('uint8'))
    out_img.save(output_path, quality=95) # Save with high quality
    
    # Immediate extraction test
    recovered_payload, rdbg = external_extract(wm_np, np_img, params_full)
    
    merged = {}
    merged.update(dbg or {})
    merged.update(rdbg or {})
    
    # Check robust payload BER
    ber = rdbg.get('payload_ber', 1.0)
    merged['embed_success'] = bool(ber < 0.01) 
    
    merged['timestamp'] = now_iso()
    merged['output_path'] = output_path
    return merged

def batch_embed(input_dir: str, output_dir: str, payload: bytes = None, params: dict = None, seed: int = 0):
    ensure_dir(output_dir)
    rows = []
    paths = sorted(Path(input_dir).glob('*'))
    paths = [p for p in paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"Embedding {len(paths)} images...")
    for path in tqdm(paths):
        out_path = Path(output_dir) / (path.stem + '_wm' + '.jpg') # Standardize to JPG
        meta = embed_image(str(path), str(out_path), payload=payload, params=params, seed=seed)
        meta['original_path'] = str(path)
        rows.append(meta)
    return rows