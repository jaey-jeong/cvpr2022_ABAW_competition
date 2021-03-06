import torch.nn as nn
import math
import torch.nn.functional as F
import networks.utils as utils
import numpy as np

__all__ = ['SENet', 'senet50']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
# This SEModule is not used.
class SEModule(nn.Module):

    def __init__(self, planes, compress_rate):
        super(SEModule, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes // compress_rate, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(planes // compress_rate, planes, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = F.avg_pool2d(module_input, kernel_size=module_input.size(2))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # SENet
        compress_rate = 16
        # self.se_block = SEModule(planes * 4, compress_rate)  # this is not used.
        self.conv4 = nn.Conv2d(planes * 4, planes * 4 // compress_rate, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv2d(planes * 4 // compress_rate, planes * 4, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        ## senet
        out2 = F.avg_pool2d(out, kernel_size=out.size(2))
        out2 = self.conv4(out2)
        out2 = self.relu(out2)
        out2 = self.conv5(out2)
        out2 = self.sigmoid(out2)
        # out2 = self.se_block.forward(out)  # not used

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out2 * out + residual
        # out = out2 + residual  # not used
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def load_from_pretrain(self, model, pretrained_checkpoint_path):
        model = utils.load_state_dict(model, pretrained_checkpoint_path)

        return model
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def senet50(**kwargs):
    """Constructs a SENet-50 model.
    """
    model = FCL(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class FCL(nn.Module) :
    num_classes = 8
    
    def __init__(self, block, layers, pretrained_checkpoint_path, num_classes, include_top=True, freeze_base=True) :
        #self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)
        super(FCL, self).__init__()
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        
        self.base = SENet(block, layers, num_classes, include_top)
        
        self.base.load_from_pretrain(self.base, pretrained_checkpoint_path)
        if freeze_base:
            for param in self.base.parameters():
                #print
                param.requires_grad = True
        self.base = nn.Sequential(*(list(self.base.children())))
        self.base = self.base[:-2]
        print("?????????")
        #self.fc = nn.Linear(512 * block.expansion, 8, bias=True)
        self.fc = nn.Linear(512 * block.expansion, 2048, bias=True)

        
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)
        #init_layer(self.fc2)
        
    def load_from_pretrain(self, model, pretrained_checkpoint_path):
        model = utils.load_state_dict(model, pretrained_checkpoint_path)

        return model
        #self.base.load_state_dict(checkpoint)

    def forward(self, x) :
        x = self.base(x)
        print(np.shape(x))
        #x= self.fc(x)
        
        #x = x.view(x.size(0), -1)
        #print(np.shape(x))
        #x = nn.Flatten()
        #x = nn.functional.log_softmax(self.fc(x), dim=0)
        #x = nn.functional.log_softmax(self.fc(x), dim=0)

        return x