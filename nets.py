'''
File name: nets.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#============================== NET ================================
# Define the model
class Net(nn.Module):
    def __init__(self, nChannel, embeddingSize, drop_prob):
        super(Net, self).__init__()
        filtsize = 5
        poolsize = 2
        stepSize = 2
        # Conv
        self.conv1 = nn.Conv2d(nChannel, 16, filtsize, stride=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(poolsize, stride=stepSize)

        self.conv2 = nn.Conv2d(16, 32, filtsize, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(poolsize, stride=stepSize)

        self.conv3 = nn.Conv2d(32, 32, filtsize, stride=1)
        self.tanh3 = nn.Tanh()

        # FC
        self.fc    = nn.Linear(672, embeddingSize)
        self.tanh4 = nn.Tanh()
        self.drop  = nn.Dropout(p=drop_prob)
        # RNN
        self.rnn   = nn.RNN(input_size=embeddingSize,
                            hidden_size=embeddingSize,
                            num_layers=1,
                            nonlinearity='tanh',
                            batch_first=True,
                            dropout=drop_prob)

    def forward_pass(self, x, hidden):
       # Convolutional Layer --1
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
       # Convolutional Layer --2
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
       # Convolutional Layer --3
        x = self.conv3(x)
        x = self.tanh3(x)
        # FC
        x = x.view(1, -1) # (B, 672)
        x = self.fc(x)
        x = self.tanh4(x)
        x = self.drop(x)
        # RNN
        x = x.unsqueeze(1) # (B, 1, embeddingSize)
        x, hidden = self.rnn(x, hidden)

        return x, hidden

    def forward(self, x1, x2, steps, hidden=None):
        left_outputs = []
        right_outputs = []
        # Number of steps
        for t in range(steps):
            # Forward pass for cam-1
            output1, hidden = self.forward_pass(x1[:, t, :, :, :], hidden)
            # Forward pass for cam-2
            output2, hidden = self.forward_pass(x2[:, t, :, :, :], hidden)
            # Append features
            left_outputs.append(output1)
            right_outputs.append(output2)

        # Temporal Pooling
        left_outputs  = torch.cat(left_outputs,   dim=1) # N * [B, 1, 128] --> [B, N, 128]
        right_outputs = torch.cat(right_outputs,  dim=1) # N * [B, 1, 128] --> [B, N, 128]
        left_output   = torch.mean(left_outputs,  dim=1) # [B, 1, 128]
        right_output  = torch.mean(right_outputs, dim=1) # [B, 1, 128]

        left_output  = left_output.view(1, -1)  # (B, 128)
        tight_output = right_output.view(1, -1) # (B, 128)

        return left_output, right_output
