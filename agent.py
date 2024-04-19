import torch as t
import torch.nn as nn
import torch.optim as optim
from loading_anim import LoadingAnim
from random import shuffle

DataFile = 'data_30k.pt'  #'data_t26946t_m.pt'
Layers = [64, 512, 512, 256, 128, 64, 3]
Epochs = 400
BatchSize = 1000
LearnRate = 0.001
ModelFile = 'model.pt'

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using {device} ")

class SnakeNet(nn.Module):
    def __init__(self, layer_sizes):
        super(SnakeNet, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            if i < len(self.layers) - 1:  # else: x = t.softmax(x, dim=1)?
                x = t.relu(x)
        return x

def train(datafile, num_epochs=100, batch_size=1000, learning_rate=0.001):
    print('Loading... ',end='')
    lal = LoadingAnim()
    lal.start()
    data = t.load(datafile)
    model = SnakeNet(Layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    prevloss = 1
    lal.stop()
    print('Now training!..')
    la = LoadingAnim(0)

    for epoch in range(1,num_epochs+1):
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
        
        if epoch%(num_epochs//4) == 0 and epoch <= 3*(num_epochs//4): # 3 times during training, rotate the states "90 degrees"
            print('Rotating states... ', end='')
            lal.start()
            data = [[t.rot90(state, 1, [0, 1]), action, reward] for state, action, reward in data]
            lal.stop()
            print("Done!")
        if epoch%(num_epochs//2) == 0 and epoch < num_epochs : # shuffles the data, reduced frequence cause it doesn't seem to matter much
            print('Shuffling data... ', end='')
            la.start()
            shuffle(data)
            la.stop()
            print("Done!")

    print('Training complete! Saving... ', end='')
    la.start()
    t.save(model.state_dict(), ModelFile) #sr27k_64-512x4-256x2-64x2-3 model_sr27k_64-512x4-256x2-128x2-64x1-3.pt
    la.stop()

class Play:
    def __init__(self, filename=ModelFile):
        print('Loading... ',end='')
        la = LoadingAnim(0)
        la.start()
        self.model = SnakeNet(Layers).to(device)
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
    train(datafile=DataFile, num_epochs=Epochs, batch_size=BatchSize, learning_rate=LearnRate )
    print(f'Model saved as {ModelFile}\nAll Finished!')