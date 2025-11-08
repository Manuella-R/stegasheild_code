import os, csv, json
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir, set_seed, now_iso, write_metadata_row
from verifier import extract_and_verify
from embedder import embed_image

# --- Configuration ---
# Updated paths to match the new structure
CONFIG = {
    'originals_dir': 'dataset/originals',
    'base_dir': 'JpegImages',                 # NEW: Base output folder
    'problem_dir': 'JpegImages/problematic',   # NEW: Problematic folder
    'output_csv': 'JpegImages/embedding_verification.csv', # NEW: CSV path
    
    'payload_bytes': b'StegaShield_v1', 
    'embed_params': {
        'payload_bytes': b'StegaShield_v1', 
        'digest_bits': 128,
        'hmac_key': b'StegaShield_key',
        'nsym': 16, 
        'N_patches': 12, 
        'bits_per_patch': 24, 
    },
    'max_retries': 3,
    'ber_threshold': 0.01,
    'seed': 1234
}
# --- End Configuration ---

FIELDNAMES = [
    'filename', 'image_path', 'original_path', 'embed_success', 
    'payload_ber', 'robust_conf', 'fragile_conf', 'fused_decision', 
    'attempts', 'notes', 'timestamp'
]

ensure_dir(CONFIG['problem_dir'])

def try_extract(path: str, original_path: str, params: dict):
    """Attempts to extract and verify the watermark."""
    try:
        res = extract_and_verify(path, original_path, params=params)
        ber = res.get('payload_ber')
        success = (ber is not None and float(ber) < CONFIG['ber_threshold'])
        return success, res
    except Exception as e:
        print(f"Error in try_extract for {path}: {e}")
        return False, {'error': str(e), 'fused_decision': 'UNCERTAIN', 'payload_ber': 1.0}

def reembed_and_check(original_path: str, out_dir: str, trial_seed: int, embed_params: dict, payload: bytes):
    """Attempts to re-embed and then immediately verify the new image."""
    try:
        name = Path(original_path).stem
        # Save to the *correct* split directory
        out_path = str(Path(out_dir) / f"{name}_reembed_{trial_seed}.jpg")
        
        meta = embed_image(
            original_path, 
            out_path, 
            payload=payload, 
            params=embed_params, 
            seed=trial_seed
        )
        
        out_img_path = meta.get('output_path', out_path)
        ok = meta.get('embed_success', False)
        
        meta.update({
            'embed_success': ok,
            'payload_ber': meta.get('payload_ber'),
            'robust_conf': meta.get('robust_conf'),
            'fragile_conf': meta.get('fragile_conf'),
            'fused_decision': meta.get('fused_decision'),
            'output_path': out_img_path
        })
        return ok, meta
    except Exception as e:
        print(f"Error in reembed_and_check for {original_path}: {e}")
        return False, {'error': str(e), 'fused_decision': 'UNCERTAIN', 'payload_ber': 1.0}

def write_header_if_not_exists(csv_path):
    if not Path(csv_path).exists():
        write_metadata_row(csv_path, {}, fieldnames=FIELDNAMES, write_header=True)

def check_all(max_retries=CONFIG['max_retries']):
    rows = []
    output_csv = CONFIG['output_csv']
    write_header_if_not_exists(output_csv)

    # NEW: Find all watermarked images in all splits
    base_path = Path(CONFIG['base_dir'])
    splits = ['train', 'val', 'test']
    all_wm_paths = []
    for split in splits:
        split_dir = base_path / split / 'watermarked'
        if split_dir.exists():
            all_wm_paths.extend(list(split_dir.glob('*')))
    
    print(f"Found {len(all_wm_paths)} watermarked images to check...")

    for p in tqdm(all_wm_paths, desc="Checking Watermarks"):
        if p.suffix.lower() not in ['.jpg','.jpeg','.png']:
            continue
        
        attempts = 0
        
        # 1. Find the corresponding original image
        original_stem = p.stem.replace('_wm','').split('_reembed_')[0]
        original_candidates = list(Path(CONFIG['originals_dir']).glob(f"{original_stem}*"))
        
        if not original_candidates:
            # Orphan file, move it
            problem_path = Path(CONFIG['problem_dir']) / p.name
            try: p.rename(problem_path)
            except Exception as e: problem_path = p
                
            row = {
                'filename': p.name, 'image_path': str(problem_path), 'original_path': '',
                'embed_success': False, 'payload_ber': None, 'robust_conf': None, 'fragile_conf': None,
                'fused_decision': 'UNCERTAIN', 'attempts': 0,
                'notes': 'no_original_found_moved_to_problematic', 'timestamp': now_iso()
            }
            rows.append(row)
            write_metadata_row(output_csv, row, fieldnames=FIELDNAMES)
            continue

        original_path = str(original_candidates[0])
        
        # 2. Try to verify the existing file
        success, res = try_extract(str(p), original_path, params=CONFIG['embed_params'])
        attempts += 1

        if success:
            # Passed on first try
            row = {
                'filename': p.name, 'image_path': str(p), 'original_path': original_path,
                'embed_success': True,
                'payload_ber': res.get('payload_ber'),
                'robust_conf': res.get('robust_conf'),
                'fragile_conf': res.get('fragile_conf'),
                'fused_decision': res.get('fused_decision'),
                'attempts': attempts,
                'notes': 'ok_on_first_check',
                'timestamp': res.get('timestamp', now_iso())
            }
            rows.append(row)
            write_metadata_row(output_csv, row, fieldnames=FIELDNAMES)
            continue

        # 3. Verification failed. Try to re-embed.
        reembed_success = False
        last_meta = res 
        
        for retry in range(1, max_retries+1):
            attempts += 1
            seed = CONFIG['seed'] + 1000 + retry
            
            # NEW: Pass the correct output directory (its current split folder)
            current_split_dir = p.parent
            
            ok, meta = reembed_and_check(
                original_path, 
                str(current_split_dir), # Pass correct dir
                trial_seed=seed, 
                embed_params=CONFIG['embed_params'], 
                payload=CONFIG['payload_bytes']
            )
            last_meta = meta
            if ok:
                reembed_success = True
                try: Path(p).unlink() # Delete old, bad file
                except Exception as e: print(f"Warning: could not delete old file {p}: {e}")
                    
                outp = Path(meta.get('output_path'))
                # Rename to standard _wm name
                standard_out = current_split_dir / f"{original_stem}_wm{outp.suffix}"
                try:
                    outp.rename(standard_out)
                except Exception as e:
                     print(f"Warning: could not rename {outp} to {standard_out}: {e}")
                     standard_out = outp
                
                row = {
                    'filename': standard_out.name, 'image_path': str(standard_out), 'original_path': original_path,
                    'embed_success': True,
                    'payload_ber': meta.get('payload_ber'),
                    'robust_conf': meta.get('robust_conf'),
                    'fragile_conf': meta.get('fragile_conf'),
                    'fused_decision': meta.get('fused_decision'),
                    'attempts': attempts,
                    'notes': f'reembedded_success_at_attempt_{retry+1}',
                    'timestamp': meta.get('timestamp', now_iso())
                }
                write_metadata_row(output_csv, row, fieldnames=FIELDNAMES)
                break 
        
        if not reembed_success:
            # All re-embed attempts failed. Move to problematic.
            problem_path = Path(CONFIG['problem_dir']) / p.name
            try: p.rename(problem_path)
            except Exception as e: problem_path = p
                
            row = {
                'filename': p.name, 'image_path': str(problem_path), 'original_path': original_path,
                'embed_success': False,
                'payload_ber': last_meta.get('payload_ber'),
                'robust_conf': last_meta.get('robust_conf'),
                'fragile_conf': last_meta.get('fragile_conf'),
                'fused_decision': last_meta.get('fused_decision'),
                'attempts': attempts,
                'notes': f'reembed_failed_after_{max_retries}_retries_moved_to_problematic',
                'timestamp': now_iso()
            }
            rows.append(row)
            write_metadata_row(output_csv, row, fieldnames=FIELDNAMES)

if __name__ == '__main__':
    set_seed(CONFIG['seed'])
    ensure_dir('dataset') # Ensure base dir exists
    ensure_dir(CONFIG['base_dir']) # Ensure JpegImages exists
    check_all()
    print("Label-check complete. See", CONFIG['output_csv'])