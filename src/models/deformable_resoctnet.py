import torch
import torch.nn as nn
import torchvision.ops
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0):

        super(BasicBlock, self).__init__()
        self.conv1 = DeformableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        if out.size(1) != residual.size(1):
            out += self.conv2(residual)
        else:
            out += residual
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.maxpool1(out)

        return out


class DeformableResOCTNet(nn.Module):

    def __init__(self, num_classes=3):

        super(DeformableResOCTNet, self).__init__()

        self.block1 = BasicBlock(in_channels=1,   out_channels=32,  kernel_size=7, padding=3)
        self.block2 = BasicBlock(in_channels=32,  out_channels=32,  kernel_size=7, padding=3)
        self.block3 = BasicBlock(in_channels=32,  out_channels=64,  kernel_size=5, padding=2)
        self.block4 = BasicBlock(in_channels=64,  out_channels=128, kernel_size=5, padding=2)
        self.block5 = BasicBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.block6 = BasicBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3)
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
        # out = out + self.get_lesion_featuremaps(out, attention_map)
        out = self.block2(out)
        # out = out + self.get_lesion_featuremaps(out, attention_map)
        out = self.block3(out)
        # out = out + self.get_lesion_featuremaps(out, attention_map)
        out = self.block4(out)
        # out = out + self.get_lesion_featuremaps(out, attention_map)
        out = self.block5(out)
        # out = out + self.get_lesion_featuremaps(out, attention_map)
        out = self.block6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.fc3(out)
        # out = self.sigmoid(self.fc3(out))

        return out
