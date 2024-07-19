import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Define the MLP Model with Multiple Hidden Layers
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Parameters
input_size = 3
hidden_sizes = [50, 30, 10]  # Example hidden layer sizes
output_size = 3

# Instantiate the model
model = MLP(input_size, hidden_sizes, output_size).to(device)

# Step 2: Create a Dataset
# Example dataset with random values
X_train = torch.randn(100, input_size)
y_train =  2 * X_train + 1 + torch.sin(X_train) 




print("Size of X_train:", X_train.size())
print("Size of y_train:", y_train.size())

# Convert to TensorDataset
train_dataset = TensorDataset(X_train, y_train)

# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

# Step 3: Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Model
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Make Predictions
# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Example test input
    test_input = torch.randn(1, input_size).to(device)
    predicted_output = model(test_input)
    print(f'Input: {test_input.cpu().numpy()}')
    print(f'Predicted Output: {predicted_output.cpu().numpy()}')
