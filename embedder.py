"""Embedder adapter. This file is now functional and aligned with 112-bit payloads."""
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# Import all logic from the core file
import watermark_core as core
from utils import set_seed, ensure_dir, now_iso


# --- Global Model Registry ---
# Loads models once (encoder/decoder) and ensures payload_len consistency with the checkpoint.
class ModelRegistry:
    def __init__(self, model_path: str = "best_residual_hybrid.pt", payload_len: int = 112):
        self.enc: Optional[torch.nn.Module] = None
        self.dec: Optional[torch.nn.Module] = None
        self.model_path = model_path
        self.payload_len = int(payload_len)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_ckpt(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                f"Please run hybrid_train.py first."
            )
        state = torch.load(self.model_path, map_location=self.device)
        ckpt_payload_len = int(state.get("payload_len", self.payload_len))
        if ckpt_payload_len != self.payload_len:
            print(
                f"[WARN] Checkpoint payload_len={ckpt_payload_len} != registry payload_len={self.payload_len}. "
                f"Using checkpoint value."
            )
            self.payload_len = ckpt_payload_len
        return state

    def get_encoder(self):
        if self.enc is None:
            state = self._load_ckpt()
            self.enc = core.Encoder(payload_len=self.payload_len).to(self.device)
            self.enc.load_state_dict(state["enc"])
            self.enc.eval()
            print(f"Loaded learned Encoder (payload_len={self.payload_len}).")
        return self.enc

    def get_decoder(self):
        if self.dec is None:
            state = self._load_ckpt()
            self.dec = core.Decoder(payload_len=self.payload_len).to(self.device)
            self.dec.load_state_dict(state["dec"])
            self.dec.eval()
            print(f"Loaded learned Decoder (payload_len={self.payload_len}).")
        return self.dec


# Instantiate the registry
# NOTE: payload_len must match the trained checkpoint and the expected payload in verifier.
REGISTRY = ModelRegistry(payload_len=112)  # 14 bytes * 8 bits/byte = 112 bits


def bytes_to_tensor(payload_bytes: bytes, payload_len: int) -> torch.Tensor:
    """Convert bytes → float tensor of bits (shape [payload_len])."""
    bitstr = core.bytes_to_bits(payload_bytes, nbits=payload_len)
    bitstr = bitstr.ljust(payload_len, "0")  # pad if too short
    bits = [float(b) for b in bitstr]
    return torch.tensor(bits, dtype=torch.float32, device=REGISTRY.device)


def _to_tensor_256(np_img: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy image → normalized CHW float32 torch tensor (256×256)."""
    img_t = torch.from_numpy(np_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(REGISTRY.device)
    img_t = F.interpolate(img_t, size=(256, 256), mode="bilinear", align_corners=False)
    return img_t


def _tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    """Convert batched CHW float32 [0,1] → HWC uint8 numpy."""
    x = x.squeeze(0).detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0).astype(np.uint8)


def external_embed(np_img: np.ndarray, payload: bytes, params: dict) -> Tuple[np.ndarray, dict]:
    """
    Implements the full HYBRID embedding pipeline.
    1) Classical SIFT/DWT/SVD (semi-fragile digest for tamper localization, with RS ECC)
    2) Learned residual embedding (robust payload bits)
    """
    # --- 1) Classical Semi-Fragile Embed (digest) ---
    digest_bits = int(params.get("digest_bits", 128))
    hmac_key = params.get("hmac_key", b"StegaShield_key")
    nsym = int(params.get("nsym", 16))  # 16 parity bytes is a good balance for capacity

    feat = core.extract_vgg_descriptor(np_img)
    digest = core.compute_hmac_digest(feat, key=hmac_key, digest_bits=digest_bits)

    try:
        classical_wm_np, locations, bits_per_patch, enc_bits, nsym_used = core.embed_digest_with_rs(
            np_img,
            digest_bits=digest,
            nsym=nsym,
            patch_size=64,
            min_patches=8,
            alpha=0.02,
        )
        meta = {
            "digest": digest,
            "encoded_len_bits": len(enc_bits),
            "bits_per_patch": bits_per_patch,
            "N_patches": len(locations),
            "nsym_used": nsym_used,
            "classical_embed_success": True,
        }
    except Exception as e:
        print(f"[WARN] Classical embed failed: {e}. Skipping classical part.")
        classical_wm_np = np_img
        meta = {"classical_embed_success": False, "error": str(e)}

    # --- 2) Learned Robust Embed (payload) ---
    enc = REGISTRY.get_encoder()
    payload_len = REGISTRY.payload_len

    # Bytes → bit tensor (1, payload_len)
    payload_t = bytes_to_tensor(payload, payload_len).unsqueeze(0)

    # 256×256 tensor for encoder
    img_256 = _to_tensor_256(classical_wm_np)

    with torch.no_grad():
        residual_256 = enc(img_256, payload_t)  # (1,3,256,256)

    # Resize residual to original size and add to classical_wm_np
    H, W = np_img.shape[:2]
    residual_full = F.interpolate(residual_256, size=(H, W), mode="bilinear", align_corners=False)

    classical_full = torch.from_numpy(classical_wm_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(REGISTRY.device)
    final_wm_t = torch.clamp(classical_full + residual_full, 0.0, 1.0)
    final_wm_np = _tensor_to_uint8_rgb(final_wm_t)

    meta["robust_payload_len"] = payload_len
    return final_wm_np, meta


# Import the extractor for embed-image smoke test
from verifier import external_extract  # safe (verifier no longer imports embedder back)


def embed_image(input_path: str, output_path: str, payload: bytes = None, params: dict = None, seed: int = 0) -> Dict:
    set_seed(seed)
    ensure_dir(os.path.dirname(output_path) or ".")
    img = Image.open(input_path).convert("RGB")
    np_img = np.array(img)

    # Resolve payload
    params = params or {}
    payload_bytes = payload or params.get("payload_bytes", b"StegaShield_v1")
    if not isinstance(payload_bytes, (bytes, bytearray)):
        payload_bytes = str(payload_bytes).encode("utf-8")
    params["payload_bytes"] = payload_bytes  # keep for verifier

    wm_np, dbg = external_embed(np_img, payload_bytes, params)

    Image.fromarray(wm_np).save(output_path, quality=95)

    # Immediate self-check (optional)
    recovered_payload, rdbg = external_extract(wm_np, np_img, params)
    merged = {}
    merged.update(dbg or {})
    merged.update(rdbg or {})

    # payload BER OK?
    ber = float(rdbg.get("payload_ber", 1.0))
    merged["embed_success"] = bool(ber < 0.01)

    merged["timestamp"] = now_iso()
    merged["output_path"] = output_path
    return merged


def batch_embed(input_dir: str, output_dir: str, payload: bytes = None, params: dict = None, seed: int = 0):
    ensure_dir(output_dir)
    rows = []
    paths = sorted(Path(input_dir).glob("*"))
    paths = [p for p in paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    print(f"Embedding {len(paths)} images...")
    for path in tqdm(paths):
        out_path = Path(output_dir) / (path.stem + "_wm.jpg")
        meta = embed_image(str(path), str(out_path), payload=payload, params=params, seed=seed)
        meta["original_path"] = str(path)
        rows.append(meta)
    return rows
