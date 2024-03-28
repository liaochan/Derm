import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, lstm_output):
        query = self.linear(lstm_output[:, -1]).unsqueeze(2)
        keys = lstm_output
       
        energy = torch.bmm(keys, query).squeeze(2)
        
        attention_weights = F.softmax(energy, dim=1).unsqueeze(1)
        context = torch.bmm(attention_weights, lstm_output).squeeze(1)
        return context


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        context = self.attention(out)
        out = self.fc(context)
        out = F.softmax(out, dim=1)
        return out


input_size = 9
hidden_size = 64
num_layers = 2
output_size = 4


model = LSTMModel(input_size, hidden_size, num_layers, output_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


input_data = np.load("data/graph_predict/all_input.npy")
target_data = np.load("data/graph_predict/all_target.npy")
target_data = target_data[:, 0, :]

x_train, x_test, y_train, y_test = train_test_split(
    input_data, target_data, test_size=0.1
)

x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

dataset = torch.utils.data.TensorDataset(x_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


def train():
    num_epochs = 30
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/graph_predict.pt")


def test():
    model.load_state_dict(torch.load("models/graph_predict.pt"))
    sum_ = 0
    for name, param in model.named_parameters():
        mul = 1
        for size_ in param.shape:
            mul *= size_
        sum_ += mul
        print("%14s : %s" % (name, param.shape))

    print("Parameters count:", sum_)
    test_output = model(x_test)
    loss = criterion(test_output, y_test)
    print(loss)


if __name__ == "__main__":
    import sys

    module = sys.argv[1]
    if module == "train":
        start = time.perf_counter()
        train()
        end = time.perf_counter()
        print(f"Train time: {end - start} seconds")
