import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from models import LensPredictorNetwork

# ── Helpers ────────────────────────────────────────────────────────────────────

def flat_to_3heads(targets):
    """label = iso_idx*9 + shutter_idx*3 + aperture_idx"""
    return targets // 9, (targets // 3) % 3, targets % 3

def heads_to_flat(iso_logits, shutter_logits, aperture_logits):
    iso_pred      = torch.argmax(iso_logits,      dim=1)
    shutter_pred  = torch.argmax(shutter_logits,  dim=1)
    aperture_pred = torch.argmax(aperture_logits, dim=1)
    return iso_pred * 9 + shutter_pred * 3 + aperture_pred

# ── Config ─────────────────────────────────────────────────────────────────────

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

BATCH_SIZE    = 64
EPOCHS        = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-3
INPUT_DIM     = 74     # from extract_features() in utils.py
TRAIN_CSV     = os.path.join(ROOT_DIR, 'data', 'lens_train.csv')
VAL_CSV       = os.path.join(ROOT_DIR, 'data', 'lens_val.csv')
IMAGE_DIR     = r'D:\Recent Advances'
SAVE_DIR      = os.path.join(SRC_DIR, 'saved_models')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Class weights ──────────────────────────────────────────────────────────────

train_df      = pd.read_csv(TRAIN_CSV)
counts        = train_df['best_param_index'].value_counts().sort_index()
weights       = 1.0 / counts.values.astype(float)
weights       = weights / weights.sum() * 27
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

most_common   = int(train_df['best_param_index'].value_counts().index[0])
majority_acc  = (train_df['best_param_index'] == most_common).mean() * 100

# ── Data ───────────────────────────────────────────────────────────────────────

print("Loading data...")
train_loader, val_loader = get_dataloader(
    TRAIN_CSV, VAL_CSV, base_img_dir=IMAGE_DIR, batch_size=BATCH_SIZE
)
print(f"  Train batches : {len(train_loader)} | Val batches: {len(val_loader)}")
print(f"  Majority class baseline : {majority_acc:.1f}%  (model must beat this)")
print(f"  Random baseline         : {100/27:.1f}%")

# ── Model ──────────────────────────────────────────────────────────────────────

print("Initializing model...")
model = LensPredictorNetwork(input_dim=INPUT_DIM).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss & Optimiser ───────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=6
)

# ── Training loop ──────────────────────────────────────────────────────────────

best_val_acc = 0.0
os.makedirs(SAVE_DIR, exist_ok=True)
print("Starting training...\n")

for epoch in range(EPOCHS):

    # ── Train ──
    model.train()
    running_loss  = 0.0
    correct_train = 0
    total_train   = 0

    for features, targets in train_loader:
        features = features.to(device)
        targets  = targets.to(device)

        iso_t, shutter_t, aperture_t = flat_to_3heads(targets)

        optimizer.zero_grad()
        iso_logits, shutter_logits, aperture_logits = model(features)

        loss = (criterion(iso_logits,      iso_t) +
                criterion(shutter_logits,  shutter_t) +
                criterion(aperture_logits, aperture_t))

        loss.backward()
        optimizer.step()

        running_loss  += loss.item()
        flat_pred      = heads_to_flat(iso_logits, shutter_logits, aperture_logits)
        correct_train += (flat_pred == targets).sum().item()
        total_train   += targets.size(0)

    train_acc = 100.0 * correct_train / total_train

    # ── Validate ──
    model.eval()
    val_loss    = 0.0
    correct_val = 0
    total_val   = 0

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets  = targets.to(device)

            iso_t, shutter_t, aperture_t = flat_to_3heads(targets)
            iso_logits, shutter_logits, aperture_logits = model(features)

            loss = (criterion(iso_logits,      iso_t) +
                    criterion(shutter_logits,  shutter_t) +
                    criterion(aperture_logits, aperture_t))

            val_loss    += loss.item()
            flat_pred    = heads_to_flat(iso_logits, shutter_logits, aperture_logits)
            correct_val += (flat_pred == targets).sum().item()
            total_val   += targets.size(0)

    val_acc = 100.0 * correct_val / total_val
    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'{SAVE_DIR}/best_lens_predictor.pth')
        print(f"  --> New best model saved! (Val Acc: {val_acc:.2f}%)")

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
print(f"Majority class baseline was: {majority_acc:.1f}%")
