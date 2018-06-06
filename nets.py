'''
File name: nets.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#============================== NET ================================
# Define the model
class Net(nn.Module):
    def __init__(self, nChannel, embeddingSize,  nPerson, drop_prob):
        super(Net, self).__init__()
        filtsize = 5
        poolsize = 2
        stepSize = 2
        # Conv
        self.conv1 = nn.Conv2d(nChannel, 16, filtsize, stride=1)
        self.tanh1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(poolsize, stride=stepSize)

        self.conv2 = nn.Conv2d(16, 32, filtsize, stride=1)
        self.tanh2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(poolsize, stride=stepSize)

        self.conv3 = nn.Conv2d(32, 32, filtsize, stride=1)
        self.tanh3 = nn.ReLU()

        # FC
        self.fc    = nn.Linear(32*7*3, embeddingSize)
        self.tanh4 = nn.ReLU()
        self.drop  = nn.Dropout(p=drop_prob)

        # Classification Layer
        self.classifierLayer = nn.Linear(embeddingSize, nPerson)
        self.log_softmax     = nn.LogSoftmax(dim=1)

        # RNN
        self.rnn   = nn.LSTM(input_size=embeddingSize,
                            hidden_size=embeddingSize,
                            num_layers=1,
                            batch_first=True)

        # Initialize weights:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_pass(self, x):
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
        x = x.view(-1, 32*7*3) # (B, 672)
        x = self.fc(x)
        x = self.tanh4(x)
        x = self.drop(x)
        # Expand dims
        x = x.unsqueeze(1) # (B, 1, embeddingSize)

        return x

    def forward_pass(self, x, hidden=None):
        outputs = []
        # CNN
        for t in range(steps):
            # Forward pass for cam-1
            output = self.forward_pass(x[:, t, :, :, :])
            # Append features
            outputs.append(output)
        # RNN
        outputs  = torch.cat(outputs, dim=1) # N * [B, 1, 128] --> [B, N, 128]
        outputs, hidden1 = self.rnn(outputs, hidden)
        # Temporal Pooling
        output   = torch.mean(outputs,  dim=1) # [B, 1, 128]
        # Classifier
        output  = output.view(-1, 128)  # (B, 128)
        id = self.classifierLayer(output)
        id = self.log_softmax(id)

        return output, id

    def forward(self, x1, x2, steps, hidden=None):
        left_output = []
        right_output = []

        #--------------------------- Left Input --------------------------------
        left_output, l_id  = forward_pass(x1)

        #--------------------------- Right Input -------------------------------
        right_output, r_id = forward_pass(x2)


        return left_output, right_output, l_id, r_id
