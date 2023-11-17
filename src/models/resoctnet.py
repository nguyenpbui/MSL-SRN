import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms.functional  as ttf

torch.autograd.set_detect_anomaly(True)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0, final=False):

        super(BasicBlock, self).__init__()
        self.kernel_size = kernel_size
        self.final = final
        self.relu = nn.ReLU(inplace=True)
        # self.ca = ChannelAttention(out)
        # self.sa = SpatialAttention(3) ### use for seed 42
        # self.alpha = 0.001
        # self.softmax = nn.Softmax2d()
        if kernel_size == 7:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            # self.relu2 = nn.ReLU(inplace=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            # self.relu3 = nn.ReLU(inplace=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        elif kernel_size == 5:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            # self.relu1 = nn.ReLU(inplace=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            # self.relu2 = nn.ReLU(inplace=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
            # self.relu1 = nn.ReLU(inplace=False)
            self.bn1 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.downsampling = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.bn_final = nn.BatchNorm2d(out_channels)
        # self.relu_final = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        if self.kernel_size == 7:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.relu(out)
            out = self.bn3(out)
        elif self.kernel_size == 5:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu(out)
            out = self.bn2(out)
        else:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.bn1(out)
        # out = self.sa(out) * out

        if out.size(1) != residual.size(1):
            # residual = self.sa(residual) * residual
            residual = self.downsampling(x)
            # out += self.downsampling(residual)
        else:
            residual = x

        # attn = self.softmax(out)

        out = out + residual # + self.alpha*attn*residual
        out = self.relu(out)
        out = self.bn_final(out)
        if not self.final:
            out = self.maxpool(out)

        return out


class ResOCTNet(nn.Module):

    def __init__(self, num_classes=3):
        super(ResOCTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # self.block1 = BasicBlock(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock(in_channels=32,  out_channels=32,  kernel_size=5)
        self.block3 = BasicBlock(in_channels=32,  out_channels=64,  kernel_size=5)
        self.block4 = BasicBlock(in_channels=64,  out_channels=128, kernel_size=3)
        self.block5 = BasicBlock(in_channels=128, out_channels=256, kernel_size=3)
        self.block6 = BasicBlock(in_channels=256, out_channels=512, kernel_size=3)

        self.avgpool = nn.AvgPool2d(kernel_size=3)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, num_classes)
        # self.dropout1 = nn.Dropout2d(p=0.5)
        
        nn.init.xavier_normal_(self.fc1.weight)


    def get_lesion_featuremaps(self, bottom, attention_map):
        bs, c, d, _ = bottom.size()
        attention_maps = torch.stack([attention_map] * c, dim=1)
        attention_maps = F.interpolate(attention_maps, size=(d, d), mode='bilinear', align_corners=True)
        attention_maps = torch.cuda.FloatTensor(attention_maps)
        return bottom * attention_maps


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

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


class ResOCTNet_(nn.Module):

    def __init__(self, num_classes=3):
        super(ResOCTNet_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.block2 = BasicBlock(in_channels=32,  out_channels=32,  kernel_size=7)
        self.block3 = BasicBlock(in_channels=32,  out_channels=64,  kernel_size=7)
        self.block4 = BasicBlock(in_channels=64,  out_channels=128, kernel_size=5)
        self.block5 = BasicBlock(in_channels=128, out_channels=256, kernel_size=5)
        self.block6 = BasicBlock(in_channels=256, out_channels=512, kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out


class ResOCTNet_SimCLR(nn.Module):
    
    def __init__(self, out_dim):
        super(ResOCTNet_SimCLR, self).__init__()
        
        self.backbone = ResOCTNet(num_classes=out_dim)
        dim_mlp = self.backbone.fc1.in_features

        self.backbone.fc1 = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                         nn.ReLU(),
                                         self.backbone.fc1)
    
    def forward(self, x):
        return self.backbone(x)


class ResOCTNet_Dual(nn.Module):
   
    def __init__(self, num_classes=3):
        super(ResOCTNet_Dual, self).__init__()
        
        self.resoctnet1 = ResOCTNet_(num_classes=3)
        self.resoctnet2 = ResOCTNet_(num_classes=3)
        # self.resoctnet3 = ResOCTNet_(num_classes=3)

        # self.fc1 = nn.Linear(1536, 256)
        self.fc = nn.Linear(1024, num_classes)
        nn.init.xavier_normal_(self.fc.weight)
        # nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x_b, x_S):
        out_b = self.resoctnet1(x_b)
        out_S = self.resoctnet2(x_S)
        # out_s = self.resoctnet3(x_s)
        # out_b = self.fc1(out_b)
        # out_S = self.fc2(out_S)
        # out_s = self.fc3(out_s)
        out = torch.cat((out_b, out_S), dim=1)
        # out = (out_b + out_S + out_s)/3
        out = self.fc(out)
        # out = self.fc2(out)

        return out


class ResOCTNet_Dual_1Input(nn.Module):
   
    def __init__(self, num_classes=3):
        super(ResOCTNet_Dual_1Input, self).__init__()
        
        self.resoctnet1 = ResOCTNet_(num_classes=3)
        self.resoctnet2 = ResOCTNet_(num_classes=3)

        self.fc = nn.Linear(1024, num_classes)
        nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x):
        out_b = self.resoctnet1(x)
        x_s = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        # x_s = ttf.resize(x, size=(224, 224))
        out_s = self.resoctnet2(x_s)
        out = torch.cat((out_b, out_s), dim=1)
        # out = (out_b + out_s)/2
        out = self.fc(out)

        return out