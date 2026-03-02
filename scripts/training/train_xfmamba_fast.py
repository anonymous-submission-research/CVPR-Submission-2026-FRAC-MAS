import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange, repeat
from sklearn.metrics import classification_report, accuracy_score

# 1. Hardware & Environment Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Auto-detect dataset path (Local vs Kaggle)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

if os.path.exists('/kaggle/input'):
    dataset_root = '/kaggle/input/balanced-augmented-dataset'
    print("Environment: Kaggle")
else:
    dataset_root = os.path.join(_PROJECT_ROOT, 'balanced_augmented_dataset')
    print("Environment: Local")

# --- Optimized Pure PyTorch Mamba Implementation ---

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        batch, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)
        
        y = []
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            h = h * dA[:, t] + x[:, t].unsqueeze(-1) * dB[:, t]
            yt = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            y.append(yt + x[:, t] * D)
            
        return torch.stack(y, dim=1)

class XFMambaClassifier(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, d_model=192, depth=8):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.num_patches = (img_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, d_model))
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                MambaBlock(d_model, d_state=16),
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((x, cls_tokens), dim=1) 
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = x + block(x)
            
        x = self.norm(x)
        return self.head(x[:, -1])

# --- Training / Verification Function ---

def train_xfmamba(epochs=10, batch_size=16, lr=5e-4):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(dataset_root, 'train')
    val_path = os.path.join(dataset_root, 'val')

    if not os.path.exists(train_path):
        print(f"Error: Path {train_path} not found.")
        return

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)
    
    loader_args = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = XFMambaClassifier(num_classes=len(train_ds.classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting Training: {len(train_ds.classes)} classes found.")
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        all_train_preds, all_train_labels = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labs in loop:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            
            _, preds = outputs.max(1)
            train_loss += loss.item()
            train_correct += preds.eq(labs).sum().item()
            
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labs.cpu().numpy())
            
            loop.set_postfix(loss=train_loss/len(train_loader), acc=100.*train_correct/len(train_ds))

        # Training Summary
        print(f"\n[Epoch {epoch+1} Train] Acc: {100.*train_correct/len(train_ds):.2f}% | Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_correct = 0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labs).sum().item()
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labs.cpu().numpy())
        
        val_acc_total = 100. * val_correct / len(val_ds)
        print(f"[Epoch {epoch+1} Valid] Acc: {val_acc_total:.2f}%")
        
        # Detailed Classification Report
        print("\n--- Validation Classification Report ---")
        print(classification_report(all_val_labels, all_val_preds, target_names=train_ds.classes))
        print("------------------------------------------")

if __name__ == "__main__":
    train_xfmamba(epochs=5) # Set to 10+ for real training
