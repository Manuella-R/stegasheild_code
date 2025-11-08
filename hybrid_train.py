import argparse
import watermark_core as core
from pathlib import Path

# --- Defaults ---
DEFAULT_IMAGE_DIR = '/content/drive/MyDrive/project_codes/models_new/JPEGImages'
MODEL_SAVE_PATH = 'best_residual_hybrid.pt'

def main(args):
    img_dir = Path(args.image_dir)
    if not img_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        print("Please ensure your dataset is available or update the --image_dir path.")
        return

    print("Starting U-Net training...")
    print(f"Source images: {args.image_dir}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Payload bits: {args.payload_len}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")

    # Uses the implementation inside watermark_core.train_residual_encoder
    core.train_residual_encoder(
        root_images=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        payload_len=args.payload_len,
        lr=args.lr,
        save_path=MODEL_SAVE_PATH,
        curriculum=True  # enable curriculum by default
    )

    print("U-Net (Encoder/Decoder) training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Learned Residual Watermark Model (Hybrid Residual Encoder/Decoder)")
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR, help='Path to directory of original training images.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--payload_len', type=int, default=112, help='Payload length in bits for the learned model. (MUST match inference)')
    args = parser.parse_args()
    main(args)
