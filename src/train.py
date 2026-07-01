import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloader
from models import LensPredictorNetwork


#########################################
# Configuration
#########################################

TRAIN_CSV = "lens_train.csv"
VAL_CSV = "lens_val.csv"

BASE_IMG_DIR = "ImageNet-ES-Diverse"

INPUT_DIM = 74          # change if your feature extractor changes

BATCH_SIZE = 64

NUM_EPOCHS = 100

LR = 1e-3

SAVE_PATH = "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#########################################
# Dataloader
#########################################

train_loader, val_loader = get_dataloader(
    TRAIN_CSV,
    VAL_CSV,
    BASE_IMG_DIR,
    batch_size=BATCH_SIZE
)

#########################################
# Model
#########################################

model = LensPredictorNetwork(INPUT_DIM).to(DEVICE)

criterion = nn.HuberLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

#########################################
# Training
#########################################

best_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    #################################
    # Train
    #################################

    model.train()

    running_loss = 0

    for features, targets in train_loader:

        features = features.to(DEVICE)

        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    #################################
    # Validation
    #################################

    model.eval()

    val_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for features, targets in val_loader:

            features = features.to(DEVICE)

            targets = targets.to(DEVICE)

            outputs = model(features)

            loss = criterion(outputs, targets)

            val_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)

            gt = torch.argmax(targets, dim=1)

            correct += (pred == gt).sum().item()

            total += targets.size(0)

    val_loss /= len(val_loader)

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Top1 Acc: {accuracy:.2f}%"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print("Model Saved.")