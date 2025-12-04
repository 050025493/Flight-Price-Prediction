import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# 1. LOAD DATA
print("Loading Processed Data...")
try:
    data = np.load('kaggle_flight_processed.npz')
except FileNotFoundError:
    print(" Error: Run 'clean_kaggle_flight.py' first!")
    exit()

X_train = torch.tensor(data['X_train'], dtype=torch.float32)
y_train = torch.tensor(data['y_train'], dtype=torch.float32)
X_test = torch.tensor(data['X_test'], dtype=torch.float32)
y_test = torch.tensor(data['y_test'], dtype=torch.float32)

input_size = X_train.shape[1] # Automatically finds input size (should be ~10)

# 2. W&B INIT
wandb.init(
    project="kaggle-flight-pro",
    config={
        "lr_start": 0.001,
        "epochs": 3000,
        "batch_size": "Full",
        "architecture": "Pro Network (256)"
    }
)

# 3. MODEL ARCHITECTURE (The Big Brain)
class FlightProNet(nn.Module):
    def __init__(self, input_dim):
        super(FlightProNet, self).__init__()
        # Layer 1
        self.layer1 = nn.Linear(input_dim, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        # Layer 2
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        # Layer 3
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        # Output
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu(self.bn2(self.layer2(x))))
        x = self.dropout3(self.relu(self.bn3(self.layer3(x))))
        x = self.output(x)
        return x

model = FlightProNet(input_size)
wandb.watch(model, log="all")

# 4. OPTIMIZER & SCHEDULER
optimizer = optim.Adam(model.parameters(), lr=0.001) # Start fast
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
criterion = nn.MSELoss()

# 5. TRAINING LOOP
epochs = 3000
best_loss = float('inf')
early_stop_count = 0        # <--- Added Counter
early_stop_patience = 500  # <--- Patience Threshold

print(f"Training on {len(X_train)} flights...")

for epoch in range(epochs):
    model.train()
    preds = model(X_train)
    loss = criterion(preds, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test)
            
            scheduler.step(test_loss)
            
            wandb.log({
                "epoch": epoch, 
                "train_loss": loss.item(), 
                "test_loss": test_loss.item(),
                "lr": optimizer.param_groups[0]['lr']
            })
            
            print(f"Epoch {epoch+1} | Test Loss: {test_loss.item():.5f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                early_stop_count = 0  # Reset Counter
                torch.save(model.state_dict(), 'best_flight_brain.pth')
            else:
                early_stop_count += 100  # Increment Counter
                
            if early_stop_count >= early_stop_patience:
                print("Early stopping triggered.")
                break


print(f" Best Test Loss: {best_loss:.5f}")
wandb.finish()