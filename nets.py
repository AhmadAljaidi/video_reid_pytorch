'''
File name: nets.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#============================== NET ================================
# Define the model
class Net(nn.Module):
    def __init__(self, nChannel, embeddingSize,  nPerson, drop_prob, batch_size):
        super(Net, self).__init__()
        filtsize = 5
        poolsize = 2
        stepSize = 2
        # Batch size:
        self.batch_size = batch_size
        # Hidden Size
        self.embeddingSize = embeddingSize

        # Conv
        self.cnn = nn.Sequential(
                   # Layer 1
                   nn.Conv2d(nChannel, 16, filtsize, stride=1, padding=2),
                   nn.Tanh(),
                   nn.MaxPool2d(poolsize, stride=stepSize),

                   # Layer 2
                   nn.Conv2d(16, 32, filtsize, stride=1, padding=2),
                   nn.Tanh(),
                   nn.MaxPool2d(poolsize, stride=stepSize),

                   # Layer 3
                   nn.Conv2d(32, 32, filtsize, stride=1, padding=2),
                   nn.Tanh(),
                   )

        # FC
        self.fc  = nn.Sequential(
                   nn.Linear(32*14*10, embeddingSize),
                   nn.Tanh(),
                   nn.Dropout(p=drop_prob),
                   )

        # Classification Layer
        self.classify = nn.Sequential(
                        nn.Linear(embeddingSize, nPerson),
                        nn.LogSoftmax(dim=1),
                        )

        # RNN
        self.rnn = nn.RNN(input_size=embeddingSize,
                          hidden_size=embeddingSize,
                          num_layers=1,
                          nonlinearity='tanh',
                          batch_first=False)

        # Initialize weights:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def net_step(self, x, hidden):
        x = self.cnn(x)
        x = x.view(-1, 32*14*10)
        x = self.fc(x)
        x = x.unsqueeze(0) # Expand dims- (1, B, 128)
        x, hidden = self.rnn(x, hidden) # RNN

        return x, hidden

    def forward_pass(self, x, steps, hidden=None):
        outputs = torch.zeros(steps, self.batch_size, self.embeddingSize).cuda()
        # CNN--> RNN
        for t in range(steps):
            # Forward pass
            output, hidden = self.net_step(x[:, t, :, :, :], hidden) # 1, B, 128
            output = output.squeeze(0)
            # Append features
            outputs[t] = output
        # Temporal Pooling
        output = nn.Parameter(torch.mean(outputs, dim=0), requires_grad=True)
        # Classifier
        id = self.classify(output)

        return output, id

    def forward(self, x1, x2, steps):
        #--------------------------- Left Input --------------------------------
        left_output, l_id  = self.forward_pass(x1, steps)

        #--------------------------- Right Input -------------------------------
        right_output, r_id = self.forward_pass(x2, steps)

        return left_output, right_output, l_id, r_id
