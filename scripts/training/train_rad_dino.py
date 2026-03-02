import argparse
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

# Ensure repo src is on path
sys.path.insert(0, os.path.abspath('src'))

# Constants
MODEL_NAME = "microsoft/rad-dino"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10

class FractureDataset(Dataset):
    def __init__(self, csv_file, image_root_dir, processor, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.processor = processor
        self.transforms = transforms
        
        # Verify if 'data/' prefix is needed
        # Check first image
        if len(self.df) > 0:
            first_path = self.df.iloc[0]['image_path']
            full_path = os.path.join(self.image_root_dir, first_path)
            if not os.path.exists(full_path):
                 # Try prepending 'data/' if root is just workspace root
                 # But the user might pass 'data/' as image_root_dir.
                 # I will assume image_root_dir is correctly passed.
                 pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        local_path = row['image_path']
        label = int(row['label'])
        
        full_path = os.path.join(self.image_root_dir, local_path)
        
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return dummy or fail? Fail for now
            raise e

        if self.transforms:
            image = self.transforms(image)

        # Processor returns dict with 'pixel_values'
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0), # Remove batch dim
            'label': torch.tensor(label, dtype=torch.long)
        }

class RadDinoClassifier(nn.Module):
    def __init__(self, num_classes, head_type='linear'):
        super(RadDinoClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.backbone.config.hidden_size
        
        if head_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # Use CLS token (index 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Rad-Dino Classifier")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for images")
    parser.add_argument("--train-csv", type=str, default="balanced_augmented_dataset/train.csv", help="Path to train CSV")
    parser.add_argument("--val-csv", type=str, default="balanced_augmented_dataset/val.csv", help="Path to val CSV")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--head-type", type=str, default="linear", choices=["linear", "mlp"], help="Classifier head type")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
             print("MPS not available because the current PyTorch install was not built with MPS enabled.")
             device = torch.device("cpu")
        else:
            device = torch.device("mps")
            print("Using Apple Metal Performance Shaders (MPS) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU. (No GPU detected)")
        
    print(f"Device: {device}")
    
    # Load processor
    print(f"Loading processor for {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    # Define Transforms
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    
    # Prepare datasets
    train_dataset = FractureDataset(args.train_csv, args.data_root, processor, transforms=train_transforms)
    val_dataset = FractureDataset(args.val_csv, args.data_root, processor, transforms=None)
    
    # Pin memory is not supported on MPS yet
    use_pin_memory = (device.type != 'cpu' and device.type != 'mps')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    
    # Check num classes
    # Assuming labels are 0-indexed integers in the CSV
    # We can infer num_classes from the max label + 1
    num_classes = train_dataset.df['label'].max() + 1
    print(f"Detected {num_classes} classes.")
    
    # Initialize model
    print(f"Initializing model with {args.head_type} head...")
    model = RadDinoClassifier(num_classes, head_type=args.head_type).to(device)
    
    # Optimizer and Loss
    # Only optimize the classifier parameters
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-3)
    # Using Label Smoothing to prevent overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step with val_loss for ReduceLROnPlateau
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Current LR: {current_lr:.6f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_rad_dino_classifier.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
             # Also good to check validation loss for stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 # reset if loss improves even if acc doesn't
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                    break
            
    print("Training complete.")

if __name__ == "__main__":
    main()
