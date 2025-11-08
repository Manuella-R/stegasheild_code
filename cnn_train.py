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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

class StegaDataset(Dataset):
    def __init__(self, metadata_csv, transform=None, use_aux=True):
        self.df = pd.read_csv(metadata_csv).fillna(0) # Fill NaNs with 0
        self.transform = transform
        self.use_aux = use_aux
        
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
        
        self.labels = self.df['class_label'].astype(int).values
        self.image_paths = self.df['image_path'].values
        
        # Prepare auxiliary features
        self.aux_features = self.df[['payload_ber', 'robust_conf', 'fragile_conf']].astype(np.float32).values
        # Normalize aux features (simple scaling)
        self.aux_features[:, 0] = self.aux_features[:, 0] * 2.0 - 1.0 # BER [0,1] -> [-1, 1]
        self.aux_features[:, 1] = self.aux_features[:, 1] * 2.0 - 1.0 # Conf [0,1] -> [-1, 1]
        self.aux_features[:, 2] = self.aux_features[:, 2] * 2.0 - 1.0 # Conf [0,1] -> [-1, 1]

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
            
        label = self.labels[idx]
        aux = self.aux_features[idx]
        
        if self.use_aux:
            return img, aux, label
        return img, label

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
    transform = transforms.Compose([
        transforms.Resize((299, 299)), # Xception input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_ds = StegaDataset(metadata_csv, transform=transform, use_aux=use_aux)
    
    # Use the splits from the CSV file
    train_df = full_ds.df[full_ds.df['dataset_split'] == 'train']
    val_df = full_ds.df[full_ds.df['dataset_split'] == 'val']
    
    train_idx = train_df.index.values
    val_idx = val_df.index.values

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training with {len(train_idx)} samples, Validating with {len(val_idx)} samples.")

    model = build_model(num_classes=3, use_aux=use_aux, num_aux_features=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds_train = []
        all_labels_train = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            inputs, auxs, labels = batch
            inputs = inputs.to(device)
            auxs = auxs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if use_aux:
                logits = model(inputs, auxs)
            else:
                logits = model(inputs)
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(all_labels_train, all_preds_train)
        train_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        
        model.eval()
        total_val_loss = 0
        all_preds_val = []
        all_labels_val = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                inputs, auxs, labels = batch
                inputs = inputs.to(device)
                auxs = auxs.to(device)
                labels = labels.to(device)
                
                if use_aux:
                    logits = model(inputs, auxs)
                else:
                    logits = model(inputs)
                    
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels_val, all_preds_val)
        val_loss = total_val_loss / len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'stegashield_cnn_final.pth')
            print(f'New best model saved with val_acc: {best_val_acc:.4f}')

        print("\nValidation Classification Report:")
        print(classification_report(all_labels_val, all_preds_val, target_names=['Original', 'Watermarked', 'Tampered'], zero_division=0))

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