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

    print("Starting Optimized training with existing architecture...")
    
    # Use existing architecture but with better training params
    core.train_residual_encoder(
        root_images=args.image_dir,
        epochs=args.epochs,          # Increase to 30-40
        batch_size=args.batch_size,  # Keep at 16-32
        payload_len=args.payload_len,
        lr=2e-4,                     # Higher LR (was 1e-4)
        save_path=MODEL_SAVE_PATH,
        curriculum=True,
        cache_to_ram=args.cache_ram
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--epochs', type=int, default=35)  # More epochs
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--payload_len', type=int, default=112)
    parser.add_argument('--cache_ram', action='store_true')
    args = parser.parse_args()
    main(args)