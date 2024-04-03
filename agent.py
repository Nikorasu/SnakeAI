import torch
import torch.nn as nn
import torch.optim as optim
#from random import shuffle

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define the neural network architecture
class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to train the neural network
def train(datafile, num_epochs=200, batch_size=128, learning_rate=0.001):
    data = torch.load(datafile)
    model = SnakeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(data), batch_size):
            batch_states, batch_actions, _, _ = zip(*data[i:i+batch_size])
            batch_states = torch.stack(batch_states).view(-1, 64).to(device, dtype=torch.float32)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(data)}")
        #shuffle(data)

    print("Training complete!")
    torch.save(model.state_dict(), "snake_model.pth")

# Function to load the model and play a turn
class Play:
    def __init__(self,filename):
        self.model = SnakeNet().to(device)
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def turn(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
        output = self.model(state)
        action = output.max(1)[1].item()
        return action

if __name__ == "__main__":
    # Train the model
    train('snakedata_t2000_37.pt')
    print("Model trained and saved as snake_model.pt")