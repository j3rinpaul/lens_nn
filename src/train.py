import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import LensPredictorNetwork 
import os

def train_model():
    # --- Configuration ---
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Initialization ---
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders('data/lens_train.csv', 'data/lens_val.csv', batch_size)
    
    print("Initializing Model...")
    model = LensPredictorNetwork(input_dim=160, output_dim=27).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Tracking the best model
    best_val_accuracy = 0.0

    print("Starting Training...")
    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (light_vectors, targets) in enumerate(train_loader):
            light_vectors, targets = light_vectors.to(device), targets.to(device)
            
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: Predict the 27 parameters
            outputs = model(light_vectors)
            
            # 3. Calculate Loss
            loss = criterion(outputs, targets)
            
            # 4. Backward pass (calculate gradients)
            loss.backward()
            
            # 5. Optimize (update weights)
            optimizer.step()
            
            # --- Metrics Tracking ---
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
        train_accuracy = 100 * correct_train / total_train
        
        # --- Validation Loop (After every epoch) ---
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for light_vectors, targets in val_loader:
                light_vectors, targets = light_vectors.to(device), targets.to(device)
                outputs = model(light_vectors)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
                
        val_accuracy = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_accuracy:.2f}%")
              
        # --- Save the best model ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_lens_predictor.pth')
            print("--> New Best Model Saved!")

if __name__ == "__main__":
    train_model()