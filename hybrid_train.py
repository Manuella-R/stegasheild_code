import argparse
import watermark_core as core
from pathlib import Path

DEFAULT_IMAGE_DIR = '/content/drive/MyDrive/project_codes/models_new/JPEGImages'
MODEL_SAVE_PATH = 'best_residual_hybrid.pt'

def main(args):
    img_dir = Path(args.image_dir)
    if not img_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        return

    save_path = args.save_path or MODEL_SAVE_PATH
    resume_from = args.resume
    if args.fresh:
        resume_from = False  # disable auto-resume inside train_residual_encoder
        sp = Path(save_path)
        if sp.exists():
            backup = sp.with_suffix(sp.suffix + ".bak")
            try:
                sp.rename(backup)
                print(f"🗑️ Existing checkpoint moved to {backup}")
            except Exception as e:
                print(f"⚠️ Could not move existing checkpoint {sp}: {e}")

    print("Starting Optimized training with existing architecture...")
    
    # Use existing architecture but with better training params
    core.train_residual_encoder(
        root_images=args.image_dir,
        epochs=args.epochs,          # Increase to 30-40
        batch_size=args.batch_size,  # Keep at 16-32
        payload_len=args.payload_len,
        lr=args.lr,
        save_path=save_path,
        curriculum=not args.no_curriculum,
        cache_to_ram=args.cache_ram,
        max_attack_strength=args.max_attack_strength,
        amp=not args.no_amp,
        resume_from=resume_from
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--epochs', type=int, default=80)  # More epochs for convergence
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--payload_len', type=int, default=112)
    parser.add_argument('--cache_ram', action='store_true')
    parser.add_argument('--max_attack_strength', type=float, default=0.45)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_curriculum', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_path', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--fresh', action='store_true', help='Ignore existing checkpoints and start from scratch')
    args = parser.parse_args()
    main(args)