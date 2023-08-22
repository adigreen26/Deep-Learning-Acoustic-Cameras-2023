# Contains the CNNFrame model definition
import torch
import torch.nn as nn
import torch.nn.functional as F


# with dropout
class CNNFrame(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.05):
        super(CNNFrame, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(dropout_prob)  # Dropout after batch normalization

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout2d(dropout_prob)  # Dropout after batch normalization

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(dropout_prob)  # Dropout after batch normalization

        self.fc_input_size = 128 * 35 * 48
        self.fc1 = nn.Linear(self.fc_input_size * 2, 512)
        self.drop_fc1 = nn.Dropout(dropout_prob)  # Dropout before the next fully connected layer

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, frame1, frame2):
        # Convolutional layers for frame1
        x1 = F.relu(self.drop1(self.bn1(self.conv1(frame1))))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x1 = F.relu(self.drop2(self.bn2(self.conv2(x1))))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x1 = F.relu(self.drop3(self.bn3(self.conv3(x1))))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)

        # Convolutional layers for frame2
        x2 = F.relu(self.drop1(self.bn1(self.conv1(frame2))))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x2 = F.relu(self.drop2(self.bn2(self.conv2(x2))))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x2 = F.relu(self.drop3(self.bn3(self.conv3(x2))))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)

        # Flatten the outputs
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)

        # Concatenate the feature vectors
        combined_out = torch.cat((x1, x2), dim=1)

        # Fully connected layers
        x = F.relu(self.drop_fc1(self.fc1(combined_out)))
        x = self.fc2(x)
        return x
