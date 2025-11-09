"""
generate_dataset.py
This script now builds the nested train/val/test dataset structure
as requested, while also creating the master metadata.csv file.
"""
import os, json, csv, random
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import ensure_dir, set_seed, write_metadata_row, now_iso
from embedder import embed_image
from attacker import apply_attacks
from verifier import extract_and_verify
import argparse

# --- Configuration ---
CONFIG = {
    'originals_dir': 'dataset/originals', # Source of original images
    'output_base_dir': 'JpegImages',     # NEW: Base output folder
    'metadata_csv': 'JpegImages/metadata.csv', # NEW: CSV path
    
    'payload_bytes': b'StegaShield_v1', # 14 bytes = 112 bits
    
    'embed_params': {
        'payload_bytes': b'StegaShield_v1', 
        'digest_bits': 128,
        'hmac_key': b'StegaShield_key',
        'nsym': 16, 
        'N_patches': 12, 
        'bits_per_patch': 24, 
    },
    'seed': 42,
    'n_jobs': 4,
    
    # NEW: Defines counts for each split
    'per_split': {
        'train': {'watermarked': 2500, 'tampered': 2500, 'unwatermarked': 1000},
        'val':   {'watermarked': 500,  'tampered': 500,  'unwatermarked': 200},
        'test':  {'watermarked': 500,  'tampered': 500,  'unwatermarked': 200}
    },
    
    # Using the expanded attack list
    'attack_presets': [
        ({'type':'jpeg', 'quality': 90}, 0.10),
        ({'type':'jpeg', 'quality': 75}, 0.10),
        ({'type':'resize', 'scale': 0.8}, 0.10),
        ({'type':'brightness_contrast', 'alpha': 1.1, 'beta': 5}, 0.05),
        ({'type':'gamma', 'gamma': 1.2}, 0.05),
        ({'type':'blur', 'radius': 1.5}, 0.05),
        ({'type':'noise', 'sigma': 5.0}, 0.05),
        ({'type':'rotate', 'angle': 3.0}, 0.05),
        ({'type':'color_jitter', 'hue': 0.05, 'saturation': 0.3}, 0.05),
        ({'type':'text_overlay', 'opacity': 0.4}, 0.05),
        ({'type':'jpeg', 'quality': 50}, 0.05),
        ({'type':'sharpen', 'strength': 0.8}, 0.05),
        ({'type':'median_blur', 'ksize': 3}, 0.05),
        ({'type':'crop', 'ratio': 0.8}, 0.05),
        ({'type':'affine', 'shear': 0.15, 'scale': 0.8}, 0.05),
        ({'type':'patch_replace', 'size': 0.15}, 0.05),
        ({'type':'salt_pepper', 'amount': 0.005}, 0.05),
    ],
    'max_embed_attempts':3,
}

FIELDNAMES = ['id','dataset_split','original_path','watermarked_path','tampered_path','class_label','attack_type','attack_params','seed','embed_success','payload_ber','robust_conf','fragile_conf','fused_decision','timestamp']

def weighted_choice(pairs):
    items, weights = zip(*pairs)
    s = sum(weights)
    if s == 0: return random.choice(items)
    probs = [w/s for w in weights]
    return random.choices(items, probs, k=1)[0]

def sample_attacks_for_count(count, presets):
    return [weighted_choice(presets) for _ in range(count)]

def make_dir_structure():
    """Creates the new train/val/test directory structure."""
    base_dir = Path(CONFIG['output_base_dir'])
    for split in CONFIG['per_split'].keys():
        ensure_dir(base_dir / split / 'watermarked')
        ensure_dir(base_dir / split / 'tampered')
        ensure_dir(base_dir / split / 'unwatermarked')
    ensure_dir(os.path.dirname(CONFIG['metadata_csv']) or '.')

def select_originals(n, used=set()):
    all_paths = sorted(Path(CONFIG['originals_dir']).glob('*'))
    candidates = [str(p) for p in all_paths if p.suffix.lower() in ['.jpg','.jpeg','.png']]
    if not candidates:
        raise RuntimeError(f"No images found in {CONFIG['originals_dir']}. Please add your dataset.")
    random.shuffle(candidates)
    chosen = []
    for c in candidates:
        if c in used:
            continue
        chosen.append(c)
        used.add(c)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        print(f'Warning: Not enough originals. Needed {n}, found {len(chosen)}. Using duplicates.')
        chosen.extend(random.choices(candidates, k=n-len(chosen)))
        used.update(chosen)
    return chosen, used

def embed_task(original_path, out_dir, payload, params, seed, attempts=3):
    # This function is the same, but 'out_dir' will now be a split-specific path
    for attempt in range(attempts):
        s = seed + attempt
        name = Path(original_path).stem
        out_path = Path(out_dir) / f"{name}_wm_a{attempt}.jpg"
        try:
            meta = embed_image(original_path, str(out_path), payload=payload, params=params, seed=s)
            meta['output_path'] = str(out_path)
            meta['attempt'] = attempt
            meta['seed_used'] = s
            if meta.get('embed_success', False):
                return True, meta
            else:
                print(f"[embed_task] attempt {attempt} embed OK, but verify (BER) failed for {original_path}.")
        except Exception as e:
            print(f"[embed_task] attempt {attempt} embed failed for {original_path}: {e}")
            continue
    return False, {'error':'embed_failed_all_attempts', 'original': original_path}

def attack_task(wm_path, out_dir, attack_spec, seed):
    # This function is the same, 'out_dir' will be a split-specific path
    try:
        rows = apply_attacks(wm_path, out_dir, [attack_spec], seed=seed)
        if len(rows) >= 1:
            return True, rows[0]
        return False, {'error':'no_output', 'wm_path': wm_path}
    except Exception as e:
        return False, {'error': str(e), 'wm_path': wm_path}

def write_header(csv_path):
    if not Path(csv_path).exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def main():
    set_seed(CONFIG['seed'])
    make_dir_structure() # Creates the new folder structure
    write_header(CONFIG['metadata_csv'])
    
    split_config = CONFIG['per_split']
    splits = ['train', 'val', 'test']
    used_originals = set()
    id_counter = 0
    random.seed(CONFIG['seed'])
    
    # --- Generation logic ---
    for split in splits:
        print(f"--- Preparing split {split} ---")
        counts = split_config.get(split)
        w_n = counts.get('watermarked')
        t_n = counts.get('tampered')
        u_n = counts.get('unwatermarked')
        
        # Define output directories for this split
        wm_dir = Path(CONFIG['output_base_dir']) / split / 'watermarked'
        tamp_dir = Path(CONFIG['output_base_dir']) / split / 'tampered'
        unwm_dir = Path(CONFIG['output_base_dir']) / split / 'unwatermarked'

        w_originals, used_originals = select_originals(w_n, used_originals)
        u_originals, used_originals = select_originals(u_n, used_originals)

        print(f"Embedding {len(w_originals)} 'watermarked' images for {split}...")
        embed_results = Parallel(n_jobs=CONFIG['n_jobs'])(
            delayed(embed_task)(p, wm_dir, CONFIG['payload_bytes'], CONFIG['embed_params'], CONFIG['seed'] + i, CONFIG['max_embed_attempts'])
            for i, p in enumerate(tqdm(w_originals, desc=f"Embedding {split}"))
        )
        
        wm_success_map = {} # Map original_path -> wm_path
        for (success, meta), original in zip(embed_results, w_originals):
            id_counter += 1
            if success:
                wm_path = meta.get('output_path')
                wm_success_map[original] = wm_path
                # Class label 1: Watermarked (benign)
                row = {'id': id_counter, 'dataset_split': split, 'original_path': original, 'watermarked_path': wm_path, 'tampered_path': '', 'class_label': 1, 'attack_type': 'none', 'attack_params': json.dumps({}), 'seed': meta.get('seed_used', CONFIG['seed']), 'embed_success': meta.get('embed_success', True), 'payload_ber': meta.get('payload_ber', None), 'robust_conf': meta.get('robust_conf', None), 'fragile_conf': meta.get('fragile_conf', None), 'fused_decision': meta.get('fused_decision', 'PASS'), 'timestamp': meta.get('timestamp', now_iso())}
            else:
                row = {'id': id_counter, 'dataset_split': split, 'original_path': original, 'watermarked_path': '', 'tampered_path': '', 'class_label': 0, 'attack_type': 'none', 'attack_params': json.dumps({}), 'seed': CONFIG['seed'], 'embed_success': False, 'payload_ber': None, 'robust_conf': None, 'fragile_conf': None, 'fused_decision': 'UNCERTAIN', 'timestamp': now_iso()}
            write_metadata_row(CONFIG['metadata_csv'], row, fieldnames=FIELDNAMES)
        
        print(f"Copying {len(u_originals)} 'unwatermarked' images for {split}...")
        for original in tqdm(u_originals, desc=f"Copying {split}"):
            id_counter += 1
            out_name = Path(original).name
            out_path = unwm_dir / out_name
            if not Path(out_path).exists():
                try:
                    Path(out_path).write_bytes(Path(original).read_bytes())
                except Exception as e:
                    print(f"Warning: could not copy {original} to {out_path}: {e}")
                    continue
            # Class label 0: Unwatermarked (Original)
            row = {'id': id_counter, 'dataset_split': split, 'original_path': original, 'watermarked_path': '', 'tampered_path': str(out_path), 'class_label': 0, 'attack_type': 'none', 'attack_params': json.dumps({}), 'seed': CONFIG['seed'], 'embed_success': False, 'payload_ber': 1.0, 'robust_conf': 0.0, 'fragile_conf': 0.0, 'fused_decision': 'PASS_NO_OWNERSHIP', 'timestamp': now_iso()}
            write_metadata_row(CONFIG['metadata_csv'], row, fieldnames=FIELDNAMES)

        # --- Tampering ---
        available_for_tamper = list(wm_success_map.items())
        if len(available_for_tamper) < t_n:
            print(f"Warning: Not enough successful watermarks for split {split}: need {t_n}, got {len(available_for_tamper)}. Using duplicates.")
            if not available_for_tamper:
                print(f"Error: No successful watermarks to tamper in split {split}. Skipping tamper gen.")
                continue
            chosen_for_tamper = random.choices(available_for_tamper, k=t_n)
        else:
            chosen_for_tamper = random.sample(available_for_tamper, t_n)
            
        attack_list = sample_attacks_for_count(t_n, CONFIG['attack_presets'])
        
        print(f"Applying {t_n} 'tampered' attacks for {split}...")
        attack_jobs = []
        for i, ((original_path, wm_path), attack_spec) in enumerate(zip(chosen_for_tamper, attack_list)):
            attack_jobs.append(
                delayed(attack_task)(wm_path, tamp_dir, attack_spec, seed=CONFIG['seed'] + id_counter + i)
            )
            
        attack_results = Parallel(n_jobs=CONFIG['n_jobs'])(
            tqdm(attack_jobs, desc=f"Attacking {split}")
        )
        
        for (success, ameta), (original_path, wm_path), attack_spec in zip(attack_results, chosen_for_tamper, attack_list):
            id_counter += 1
            if success:
                out_path = ameta.get('output')
                # VERIFY the tampered image
                v = extract_and_verify(out_path, original_path, params=CONFIG['embed_params'])
                # Class label 2: Tampered
                row = {'id': id_counter, 'dataset_split': split, 'original_path': original_path, 'watermarked_path': wm_path, 'tampered_path': out_path, 'class_label': 2, 'attack_type': attack_spec.get('type'), 'attack_params': json.dumps({k:v for k,v in attack_spec.items() if k!='type'}), 'seed': CONFIG['seed'], 'embed_success': v.get('payload_ber') is not None and float(v.get('payload_ber',1.0)) < 0.01, 'payload_ber': v.get('payload_ber'), 'robust_conf': v.get('robust_conf'), 'fragile_conf': v.get('fragile_conf'), 'fused_decision': v.get('fused_decision', 'UNCERTAIN'), 'timestamp': v.get('timestamp', now_iso())}
            else:
                row = {'id': id_counter, 'dataset_split': split, 'original_path': original_path, 'watermarked_path': wm_path, 'tampered_path': '', 'class_label': 2, 'attack_type': attack_spec.get('type'), 'attack_params': json.dumps({k:v for k,v in attack_spec.items() if k!='type'}), 'seed': CONFIG['seed'], 'embed_success': False, 'payload_ber': None, 'robust_conf': None, 'fragile_conf': None, 'fused_decision': 'TAMPER', 'timestamp': now_iso()}
            write_metadata_row(CONFIG['metadata_csv'], row, fieldnames=FIELDNAMES)
            
    print("Dataset generation complete. Metadata at:", CONFIG['metadata_csv'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the StegaShield dataset.")
    parser.add_argument('--originals', type=str, default=CONFIG['originals_dir'], help='Path to original images.')
    parser.add_argument('--jobs', type=int, default=CONFIG['n_jobs'], help='Number of parallel jobs.')
    
    args = parser.parse_args()
    CONFIG['originals_dir'] = args.originals
    CONFIG['n_jobs'] = args.jobs
    
    main()