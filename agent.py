import torch as t
import torch.nn as nn
import torch.optim as optim
from loading_anim import LoadingAnim
from random import shuffle

# Check if CUDA is available
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using {device} ")

# Define the neural network architecture
class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        x = t.relu(self.fc1(x))
        x = t.relu(self.fc2(x))
        x = t.relu(self.fc3(x))
        x = t.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Function to train the neural network
def train(datafile, num_epochs=300, batch_size=1000, learning_rate=0.001):
    print('Loading... ',end='')
    lal = LoadingAnim()
    lal.start()
    data = t.load(datafile)
    model = SnakeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    prevloss = 1
    lal.stop()
    print('Now training!..')
    la = LoadingAnim(0)

    for epoch in range(1,num_epochs):
        la.start()
        running_loss = 0.0
        for i in range(0, len(data), batch_size):
            batch_states, batch_actions, _ = zip(*data[i:i+batch_size]) #, _
            batch_states = t.stack(batch_states).view(-1, 64).to(device, dtype=t.float32)
            batch_actions = t.tensor(batch_actions, dtype=t.long).to(device)
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        la.stop()
        darken = '\x1b[2m' if prevloss < running_loss / len(data) else ''
        print(f"\rEpoch {epoch}, Loss: {darken}{running_loss / len(data)}\x1b[0m")
        prevloss = running_loss / len(data)
        
        if epoch % (num_epochs//4) == 0: # 3 times during training, rotate the states "90 degrees"
            print('Rotating states... ', end='')
            lal.start()
            data = [[t.rot90(state, -1, [0, 1]), action, reward] for state, action, reward in data]
            lal.stop()
            print()
        elif epoch % 10 == 0: # otherwise every 10 epochs, shuffle the data
            print('Shuffling data... ', end='')
            la.start()
            shuffle(data)
            la.stop()
            print()

    print("Training complete!")
    t.save(model.state_dict(), "model_sr27k_64-512x4-256x2-64x2-3.pt")  #"snakemodel_7000_64-256x2-512x2-256x2-64x2-3.pth"

# Function to load the model and play a turn
class Play:
    def __init__(self,filename):
        print('Loading... ',end='')
        la = LoadingAnim(0)
        la.start()
        self.model = SnakeNet().to(device)
        self.model.load_state_dict(t.load(filename))
        self.model.eval()
        la.stop()
        print('Game On!')

    def turn(self, state):
        state = state.clone().detach().view(1, -1).to(device, dtype=t.float32) #t.tensor(state, dtype=t.float32, device=device)
        output = self.model(state)
        action = output.max(1)[1].item()
        return action

if __name__ == "__main__":
    # Train the model
    train('data_t26946t_m.pt') #'snakedata_t9946t_38.pt'
    print("Model trained and saved!")