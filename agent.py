import torch
import torch.nn as nn
import torch.optim as optim
from loading_anim import LoadingAnim
#from random import shuffle

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.. ")

# Define the neural network architecture
class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 3)
        #self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to train the neural network
def train(datafile, num_epochs=300, batch_size=128, learning_rate=0.001):
    print('Loading... ',end='')
    la = LoadingAnim()
    la.start()
    data = torch.load(datafile)
    model = SnakeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    la.stop()
    print('Now training!..')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(data), batch_size):
            batch_states, batch_actions, _ = zip(*data[i:i+batch_size]) #, _
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
    torch.save(model.state_dict(), "snakemodel_300x17k_64-512x4-256x2-3.pt")  #"snakemodel_7000_64-256x2-512x2-256x2-64x2-3.pth"

# Function to load the model and play a turn
class Play:
    def __init__(self,filename):
        print('Loading... ',end='')
        la = LoadingAnim(False)
        la.start()
        self.model = SnakeNet().to(device)
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        la.stop()
        print('Game On!')

    def turn(self, state):
        state = state.clone().detach().view(1, -1).to(device, dtype=torch.float32) #torch.tensor(state, dtype=torch.float32, device=device)
        output = self.model(state)
        action = output.max(1)[1].item()
        return action

if __name__ == "__main__":
    # Train the model
    train('snakedata_t16946t_m.pt') #'snakedata_t9946t_38.pt'
    print("Model trained and saved!")