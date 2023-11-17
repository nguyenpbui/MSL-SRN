import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



class BasicBlock_OCTNet(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0):

        super(BasicBlock_OCTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.maxpool1(out)

        return out


class GlobalOCTNet(nn.Module):

    def __init__(self):

        super(GlobalOCTNet, self).__init__()

        self.block1 = BasicBlock_OCTNet(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock_OCTNet(in_channels=32,  out_channels=32,  kernel_size=7, padding=3)
        self.block3 = BasicBlock_OCTNet(in_channels=32,  out_channels=64,  kernel_size=5, padding=2)
        self.block4 = BasicBlock_OCTNet(in_channels=64,  out_channels=128, kernel_size=5, padding=2)
        self.block5 = BasicBlock_OCTNet(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.block6 = BasicBlock_OCTNet(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3)


    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out


class LocalOCTNet(nn.Module):

    def __init__(self):

        super(LocalOCTNet, self).__init__()

        self.block1 = BasicBlock_OCTNet(in_channels=1,   out_channels=32,  kernel_size=3, padding=1)
        self.block2 = BasicBlock_OCTNet(in_channels=32,  out_channels=32,  kernel_size=3, padding=1)
        self.block3 = BasicBlock_OCTNet(in_channels=32,  out_channels=64,  kernel_size=3, padding=1)
        self.block4 = BasicBlock_OCTNet(in_channels=64,  out_channels=64,  kernel_size=3, padding=1)
        self.block5 = BasicBlock_OCTNet(in_channels=64, out_channels=128,  kernel_size=3, padding=1)
        # self.block6 = BasicBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3)


    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        # out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out



class DualBranchOCTNet(nn.Module):

    def __init__(self, num_classes=3):

        super(DualBranchOCTNet, self).__init__()

        self.globalnet = GlobalOCTNet()
        self.localnet = LocalOCTNet()

        self.fc1 = nn.Linear(512, 128)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(32, num_classes)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, x):

        out_g = self.globalnet(x)
        out_l11 = self.localnet(x[:,:,:112,:112])
        out_l12 = self.localnet(x[:,:,:112,112:])
        out_l21 = self.localnet(x[:,:,112:,:112])
        out_l22 = self.localnet(x[:,:,112:,112:])
        out_l = torch.cat((out_l11, out_l12, out_l21, out_l22), dim=1)
        out = (out_g + out_l)/2
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)

        return out


#############################START ResOCTNet#######################################33

class BasicBlock_ResOCTNet(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0, final_bl=False):

        super(BasicBlock_ResOCTNet, self).__init__()
        self.kernel_size = kernel_size
        self.final_bl = final_bl
        if kernel_size == 7:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.relu2 = nn.ReLU(inplace=True)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.relu3 = nn.ReLU(inplace=True)
            self.bn3 = nn.BatchNorm2d(out_channels)
        elif kernel_size == 5:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.relu2 = nn.ReLU(inplace=True)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.relu1 = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.downsampling = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.relu_final = nn.ReLU(inplace=True)
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        if self.kernel_size == 7:
            out = self.bn1(self.relu1(self.conv1(x)))
            out = self.bn2(self.relu2(self.conv2(out)))
            out = self.bn3(self.relu3(self.conv3(out)))
        elif self.kernel_size == 5:
            out = self.bn1(self.relu1(self.conv1(x)))
            out = self.bn2(self.relu2(self.conv2(out)))
        else:
            out = self.bn1(self.relu1(self.conv1(x)))

        if out.size(1) != residual.size(1):
            out += self.bn_final(self.downsampling(residual))
        else:
            out += residual

        out = self.relu_final(out)
        out = self.bn_final(out)
        if not self.final_bl:
            out = self.maxpool(out)

        return out


class GlobalResOCTNet(nn.Module):

    def __init__(self):

        super(GlobalResOCTNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # self.block1 = BasicBlock_ResOCTNet(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock_ResOCTNet(in_channels=32,  out_channels=32,  kernel_size=7, padding=3)
        self.block3 = BasicBlock_ResOCTNet(in_channels=32,  out_channels=64,  kernel_size=5, padding=2)
        self.block4 = BasicBlock_ResOCTNet(in_channels=64,  out_channels=128, kernel_size=5, padding=2)
        self.block5 = BasicBlock_ResOCTNet(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.block6 = BasicBlock_ResOCTNet(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3)


    def forward(self, x):

        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.maxpool1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out


class LocalResOCTNet(nn.Module):

    def __init__(self):

        super(LocalResOCTNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # self.block1 = BasicBlock_ResOCTNet(in_channels=1,   out_channels=32,  kernel_size=3, padding=1)
        self.block2 = BasicBlock_ResOCTNet(in_channels=32,  out_channels=32,  kernel_size=5, padding=2)
        self.block3 = BasicBlock_ResOCTNet(in_channels=32,  out_channels=64,  kernel_size=3, padding=1)
        self.block4 = BasicBlock_ResOCTNet(in_channels=64,  out_channels=64,  kernel_size=3, padding=1)
        self.block5 = BasicBlock_ResOCTNet(in_channels=64, out_channels=128,  kernel_size=3, padding=1)
        self.block6 = BasicBlock_ResOCTNet(in_channels=128, out_channels=128, kernel_size=3, padding=1, final_bl=True)

        self.avgpool = nn.AvgPool2d(kernel_size=3)


    def forward(self, x):

        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.maxpool1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out



class DualBranchResOCTNet(nn.Module):

    def __init__(self, num_classes=3):

        super(DualBranchResOCTNet, self).__init__()

        self.globalnet = GlobalResOCTNet()
        self.localnet = LocalResOCTNet()

        self.fc_g = nn.Linear(512, num_classes)
        self.fc_l = nn.Linear(512, num_classes)
        # self.dropout1 = nn.Dropout2d(p=0.5)
        # self.fc2 = nn.Linear(128, 32)
        # self.dropout2 = nn.Dropout2d(p=0.5)
        # self.fc3 = nn.Linear(32, num_classes)
        
        nn.init.xavier_normal_(self.fc_g.weight)
        nn.init.xavier_normal_(self.fc_l.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, x):

        out_g = self.globalnet(x)
        out_l11 = self.localnet(x[:,:,:112,:112])
        out_l12 = self.localnet(x[:,:,:112,112:])
        out_l21 = self.localnet(x[:,:,112:,:112])
        out_l22 = self.localnet(x[:,:,112:,112:])
        out_l = torch.cat((out_l11, out_l12, out_l21, out_l22), dim=1)
        # out = (out_g + out_l)/2
        # out = self.dropout1(self.fc1(out))
        # out = self.dropout2(self.fc2(out))
        out_l = self.fc_l(out_l)
        out_g = self.fc_g(out_g)

        return out_g, out_l
