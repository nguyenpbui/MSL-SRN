import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.downsampling = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)

        # if out.size(1) != residual.size(1):
        #     residual = self.downsampling(x)
        # else:
        #     residual = x

        # out = out + residual
        out = self.maxpool1(out)

        return out


class OCTNet(nn.Module):

    def __init__(self, num_classes=3):

        super(OCTNet, self).__init__()

        self.block1 = BasicBlock(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock(in_channels=32,  out_channels=32,  kernel_size=7, padding=3)
        self.block3 = BasicBlock(in_channels=32,  out_channels=64,  kernel_size=5, padding=2)
        self.block4 = BasicBlock(in_channels=64,  out_channels=128, kernel_size=5, padding=2)
        self.block5 = BasicBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.block6 = BasicBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(3)
        self.fc1 = nn.Linear(512, 128)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(32, num_classes)
        # self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)
        # out = self.sigmoid(self.fc3(out))

        return out


class OCTNet_(nn.Module):

    def __init__(self, num_classes=3):

        super(OCTNet_, self).__init__()

        self.block1 = BasicBlock(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock(in_channels=32,  out_channels=32,  kernel_size=7, padding=3)
        self.block3 = BasicBlock(in_channels=32,  out_channels=64,  kernel_size=5, padding=2)
        self.block4 = BasicBlock(in_channels=64,  out_channels=128, kernel_size=5, padding=2)
        self.block5 = BasicBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.block6 = BasicBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 128)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(32, num_classes)
        # self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


    def get_lesion_featuremaps(self, bottom, attention_map):
        bs, c, d, _ = bottom.size()
        attention_maps = torch.stack([attention_map] * c, dim=1)
        attention_maps = F.interpolate(attention_maps, size=(d, d), mode='bilinear')
        attention_maps = torch.cuda.FloatTensor(attention_maps)
        return bottom * attention_maps


    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout1(self.fc1(out))
        # out = self.dropout2(self.fc2(out))
        # out = self.fc3(out)
        # out = self.sigmoid(self.fc3(out))

        return out


class OCTNet_Dual(nn.Module):
    def __init__(self, num_classes=3):
        super(OCTNet_Dual, self).__init__()
        
        self.resoctnet1 = OCTNet_(num_classes=3)
        self.resoctnet2 = OCTNet_(num_classes=3)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.dropout2 = nn.Dropout2d(p=0.5)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, x_b, x_s):
        out_b = self.resoctnet1(x_b)
        out_s = self.resoctnet2(x_s)
        out = torch.cat((out_b, out_s), dim=1)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)

        return out