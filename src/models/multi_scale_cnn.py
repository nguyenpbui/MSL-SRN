import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class MultiScaleCNN(nn.Module):

    def __init__(self, num_classes=3):

        super(MultiScaleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        
        self.conv5_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=7)

        self.fc1 = nn.Linear(4800, 256)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, x):

        out = self.maxpool(self.conv1(x))
        out = self.maxpool(self.conv2(out))
        out = self.maxpool(self.conv3(out))
        out = self.maxpool(self.conv4(out))

        out1 = self.conv5_1(out)
        out3 = self.conv5_3(out)
        out5 = self.conv5_5(out)

        out = torch.cat((out1, out3, out5), dim=1)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)
        # out = self.sigmoid(self.fc3(out))

        return out
