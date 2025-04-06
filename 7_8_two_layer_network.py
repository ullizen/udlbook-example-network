import torch, torch.nn as nn # torch.nn is used to train and build neural networks
from torch.utils.data import TensorDataset, DataLoader # Utility class that wraps tensors into a dataset object
from torch.optim.lr_scheduler import StepLR # Step Learning Rate (learning rate changes as a step function)

# Input size, hidden layer size, output size
D_i, D_k, D_o = 10, 40, 5

# Define a model with two hidden layers. Sequential lets you stack layers together in order
model = nn.Sequential(
    nn.Linear(D_i, D_k), # b + wx
    nn.ReLU(),
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

# Initialise parameters with He initialisation
def weight_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight) # _ at the end of the function name means it is an in-place operation
        layer_in.bias.data.fill_(0.0)

model.apply(weight_init)

# Least squares loss function
criterion = nn.MSELoss() 

# SGD optimiser
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Setup learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

x = torch.randn(100, D_i) # 100 samples of input data
y = torch.randn(100, D_o) # 100 samples of output data

data_loader = DataLoader(TensorDataset(x,y), batch_size=10)

for epoch in range(100):
    epoch_loss = 0.0

    for i, data in enumerate(data_loader):
        x_batch, y_batch = data
        optimizer.zero_grad() # Zero the gradients before the backward pass
        pred = model(x_batch) # Forward pass - make prediction based on input batch
        loss = criterion(pred, y_batch) # Calculate loss based on prediction and true value
        loss.backward() # Backward pass
        optimizer.step() # Update weights based on gradients
        epoch_loss += loss.item()

    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')

    scheduler.step() # Update learning rate