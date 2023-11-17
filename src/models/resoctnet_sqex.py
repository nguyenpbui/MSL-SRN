import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class SEBlock(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SEBlock, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.fc1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        # y = y.permute(0, 2, 3, 1)
        # y = self.nonlin1(self.linear1(y))
        # y = self.nonlin2(self.linear2(y))
        # y = y.permute(0, 3, 1, 2)
        # y = x * y
        # return y

        batch_size, num_channels, H, W = x.size()
        # Average along each channel
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(x, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sqex = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.sqex(out)
        if out.size(1) != residual.size(1):
            out += self.conv2(residual)
        else:
            out += residual
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)

        return out


class ResOCTNet_SE(nn.Module):

    def __init__(self, num_classes=3):

        super(ResOCTNet_SE, self).__init__()

        self.block1 = BasicBlock(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock(in_channels=32,  out_channels=32,  kernel_size=7, padding=3)
        self.block3 = BasicBlock(in_channels=32,  out_channels=64,  kernel_size=5, padding=2)
        self.block4 = BasicBlock(in_channels=64,  out_channels=128, kernel_size=5, padding=2)
        self.block5 = BasicBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.block6 = BasicBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.fc1 = nn.Linear(512, 3)
        # self.dropout1 = nn.Dropout2d(p=0.5)
        # self.fc2 = nn.Linear(128, 32)
        # self.dropout2 = nn.Dropout2d(p=0.5)
        # self.fc3 = nn.Linear(32, num_classes)
        # self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)


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
        out = self.fc1(out)
        # out = self.sigmoid(self.fc3(out))

        return out
