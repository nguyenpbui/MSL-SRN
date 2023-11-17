import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class CNN0(nn.Module):

    def __init__(self, num_classes=3):

        super(CNN0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(12)
        self.bn4 = nn.BatchNorm2d(24)
        self.bn5 = nn.BatchNorm2d(48)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1536, 15)
        self.dropout1 = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, x):

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))
        out = self.maxpool(self.relu(self.bn3(self.conv3(out))))
        out = self.maxpool(self.relu(self.bn4(self.conv4(out))))
        out = self.maxpool(self.relu(self.bn5(self.conv5(out))))

        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))

        return out

class CNN1(nn.Module):

    def __init__(self, num_classes=3):

        super(CNN1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(24)
        self.bn4 = nn.BatchNorm2d(48)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1536, 15)
        self.dropout1 = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, x):

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))
        out = self.maxpool(self.relu(self.bn3(self.conv3(out))))
        out = self.maxpool(self.relu(self.bn4(self.conv4(out))))

        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))

        return out

class CNN2(nn.Module):

    def __init__(self, num_classes=3):

        super(CNN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1536, 15)
        self.dropout1 = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, x):

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))
        out = self.maxpool(self.relu(self.bn3(self.conv3(out))))

        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))

        return out

class CNN3(nn.Module):

    def __init__(self, num_classes=3):

        super(CNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(1536, 15)
        self.dropout1 = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, x):

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))

        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))

        return out

class MDFF(nn.Module):
    def __init__(self, num_classes=3):
        super(MDFF, self).__init__()

        self.cnn0 = CNN0()
        self.cnn1 = CNN1()
        self.cnn2 = CNN2()
        self.cnn3 = CNN3()

        self.fc = nn.Linear(60, num_classes)
        self.dropout = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x0, x1, x2, x3):
        out0 = self.cnn0(x0)
        out1 = self.cnn1(x1)
        out2 = self.cnn2(x2)
        out3 = self.cnn3(x3)

        out = torch.cat((out0, out1, out2, out3), dim=1)
        out = self.dropout(self.fc(out))

        return out
    
class MDFF_SingleInput(nn.Module):
    def __init__(self, num_classes=3):
        super(MDFF_SingleInput, self).__init__()

        self.cnn0 = CNN0()
        self.cnn1 = CNN1()
        self.cnn2 = CNN2()
        self.cnn3 = CNN3()

        self.fc = nn.Linear(60, num_classes)
        self.dropout = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        out0 = self.cnn0(x)
        x1 = F.interpolate(x, size=(64, 128), mode='bilinear', align_corners=True)
        out1 = self.cnn1(x1)
        x2 = F.interpolate(x, size=(32, 64), mode='bilinear', align_corners=True)
        out2 = self.cnn2(x2)
        x3 = F.interpolate(x, size=(16, 32), mode='bilinear', align_corners=True)
        out3 = self.cnn3(x3)

        out = torch.cat((out0, out1, out2, out3), dim=1)
        out = self.dropout(self.fc(out))

        return out