"""
Training script (Xception preferred via timm).
This script is now MODIFIED to accept auxiliary features (payload_ber, robust_conf, fragile_conf)
which will significantly boost accuracy, likely achieving your 85-90% goal.
"""
import os, argparse, random
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class StegaDataset(Dataset):
    def __init__(self, metadata_csv, transform=None, use_aux=True, split=None):
        self.df = pd.read_csv(metadata_csv).fillna(0) # Fill NaNs with 0
        self.transform = transform
        self.use_aux = use_aux
        if split is not None:
            self.df = self.df[self.df['dataset_split'] == split]
        
        # Define the path to use based on the class label
        def get_path(row):
            label = int(row.get('class_label', 0))
            if label == 2: # Tampered
                return row.get('tampered_path')
            elif label == 1: # Watermarked
                return row.get('watermarked_path')
            else: # Label 0 (Unwatermarked)
                return row.get('tampered_path') # In generate_dataset, unwatermarked originals are copied to 'tampered_path'
        
        self.df['image_path'] = self.df.apply(get_path, axis=1)
        # Filter out rows where the image path is missing
        self.df = self.df[self.df['image_path'].apply(lambda x: isinstance(x, str) and len(x) > 0 and Path(x).exists())]
        # Reset index so DataLoader indices are dense [0, N)
        self.df = self.df.reset_index(drop=True)
        
        self.labels = self.df['class_label'].astype(int).values
        self.image_paths = self.df['image_path'].values
        
        # Prepare auxiliary features
        aux_cols = ['payload_ber', 'robust_conf', 'fragile_conf']
        if self.use_aux:
            self.aux_features = self.df[aux_cols].astype(np.float32).values
        else:
            # still keep placeholder zeros so dataloader returns consistent tuple
            self.aux_features = np.zeros((len(self.df), len(aux_cols)), dtype=np.float32)
        # Normalize aux features (simple scaling)
        self.aux_features[:, 0] = self.aux_features[:, 0] * 2.0 - 1.0 # BER [0,1] -> [-1, 1]
        self.aux_features[:, 1] = self.aux_features[:, 1] * 2.0 - 1.0 # Conf [0,1] -> [-1, 1]
        self.aux_features[:, 2] = self.aux_features[:, 2] * 2.0 - 1.0 # Conf [0,1] -> [-1, 1]
        self.aux_features = torch.from_numpy(self.aux_features)

        print(f"Loaded {len(self.df)} valid samples from {metadata_csv}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}, returning dummy data. Error: {e}")
            img = torch.zeros(3, 299, 299) # Dummy image
            
        label = int(self.labels[idx])
        aux = self.aux_features[idx]
        
        return img, aux, label

class HybridXceptionModel(nn.Module):
    def __init__(self, num_classes=3, num_aux_features=3):
        super().__init__()
        # Load pre-trained Xception model
        self.backbone = timm.create_model('xception', pretrained=True, num_classes=0) # num_classes=0 returns features
        
        # Get feature dimension from the backbone
        self.num_cnn_features = self.backbone.num_features
        
        # Create a new classifier head that combines CNN features + Aux features
        self.classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features + num_aux_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, aux):
        # Get image features
        cnn_features = self.backbone(img) # Shape: (batch_size, num_cnn_features)
        
        # Concatenate image features and auxiliary features
        combined_features = torch.cat([cnn_features, aux], dim=1)
        
        # Pass through the new classifier
        logits = self.classifier(combined_features)
        return logits


def build_model(num_classes=3, use_aux=True, num_aux_features=3):
    if use_aux:
        print("Building Hybrid Xception model (CNN + Aux features)")
        model = HybridXceptionModel(num_classes=num_classes, num_aux_features=num_aux_features)
    else:
        print("Building standard Xception model (CNN only)")
        try:
            model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        except Exception:
            print("Xception failed, falling back to EfficientNet-B3")
            model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
    return model

def train(metadata_csv, epochs=10, batch_size=16, lr=1e-4, use_aux=True):
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop((299, 299), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = StegaDataset(metadata_csv, transform=train_transform, use_aux=use_aux, split='train')
    val_ds = StegaDataset(metadata_csv, transform=eval_transform, use_aux=use_aux, split='val')
    
    if len(train_ds) == 0:
        raise ValueError(f"No training samples found in metadata CSV: {metadata_csv}")
    
    print(f"Training with {len(train_ds)} samples, Validating with {len(val_ds)} samples.")
    
    # Balance classes with weighted sampling
    train_labels = train_ds.labels
    class_sample_count = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / np.clip(class_sample_count, a_min=1, a_max=None)
    sample_weights = class_weights[train_labels]
    train_sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)

    num_workers = min(8, os.cpu_count() or 0)
    loader_common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_common_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    else:
        loader_common_kwargs.update(dict(persistent_workers=False))
    
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=False, **loader_common_kwargs)
    
    has_val = len(val_ds) > 0
    if has_val:
        val_loader = DataLoader(val_ds, shuffle=False, **loader_common_kwargs)
    else:
        val_loader = None
        print("Warning: no validation samples found; validation metrics will be skipped.")

    model = build_model(num_classes=3, use_aux=use_aux, num_aux_features=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds_train = []
        all_labels_train = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            inputs, auxs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            auxs = auxs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == 'cuda')):
                if use_aux:
                    logits = model(inputs, auxs)
                else:
                    logits = model(inputs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(all_labels_train, all_preds_train)
        train_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        
        if has_val:
            model.eval()
            total_val_loss = 0
            all_preds_val = []
            all_labels_val = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                    inputs, auxs, labels = batch
                    inputs = inputs.to(device, non_blocking=True)
                    auxs = auxs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast(enabled=(device.type == 'cuda')):
                        if use_aux:
                            logits = model(inputs, auxs)
                        else:
                            logits = model(inputs)
                        loss = criterion(logits, labels)
                    
                    total_val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    all_preds_val.extend(preds.cpu().numpy())
                    all_labels_val.extend(labels.cpu().numpy())
            
            val_acc = accuracy_score(all_labels_val, all_preds_val) if len(all_labels_val) else 0.0
            val_loss = total_val_loss / max(1, len(val_loader))
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'stegashield_cnn_final.pth')
                print(f'New best model saved with val_acc: {best_val_acc:.4f}')

            print("\nValidation Classification Report:")
            print(classification_report(all_labels_val, all_preds_val, target_names=['Original', 'Watermarked', 'Tampered'], zero_division=0))
        else:
            scheduler.step(train_loss)

    print('Training complete. Best model saved to stegashield_cnn_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default='dataset/metadata.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32) # Increased batch size
    parser.add_argument('--no_aux', action='store_true', help="Disable auxiliary features")
    args = parser.parse_args()
    
    train(args.metadata, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, use_aux=not args.no_aux)