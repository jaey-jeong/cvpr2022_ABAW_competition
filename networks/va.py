from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from torchinfo import summary
import networks.resnet as ResNet
import networks.senet as SeNet
from networks.dan_va_model import DAN
import numpy as np
N_IDENTITY = 8631 
class DAN_ab(nn.Module):
    def __init__(self, pretrained, num_class, num_head=4, ):
        super(DAN_ab, self).__init__()
        
       
        
        include_top = True 
        resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_ft_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        self.num_class = num_class
        #resnet = SeNet.senet50(pretrained_checkpoint_path="./models/senet50_scratch_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        #print(self.features)
        #print("콩콩")
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        #self.fc1 = 
        self.model = DAN(num_class=8, num_head=4)
        if pretrained == "Yes" :
            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.dan=nn.Sequential(*list(self.model.children()))[:-2]
        self.fc = nn.Linear(512, self.num_class)
        self.bn = nn.BatchNorm1d(self.num_class)
        
        #print()
        #print('dan',self.dan)

    def forward(self, x):
        x=self.features(x)
        # print(np.shape(x))
        x=self.conv1x1_1(x) 
        # print(np.shape(x))
        x=self.conv1x1_2(x)

        #print(np.shape(x))
        x, heads = self.model(x)
        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        
       
        return out, x, heads