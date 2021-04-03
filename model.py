from efficientnet_pytorch import EfficientNet
import torchvision

import torch 
import torch.nn as nn

arch = 'efficientnet-b3'

class DRModel(nn.Module):
    def __init__(self, arch = arch , pre=True):
        super(DRModel,self).__init__()
        m = EfficientNet.from_pretrained(arch ) if pre else EfficientNet.from_name(arch)
        conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #w = (m.layer0.conv1.weight.sum(1)).unsqueeze(1)
        #conv.weight = nn.Parameter(w)
        #self.conv1=conv
        self.base = m
        self.base._fc = nn.Linear(1536,1,bias = True)
        
        #nc = self.layer4[-1].se_module.fc2.out_channels       # changes as per architecture
        #self.head = Head(nc,1)
#         self.head1 = Head(nc,n[0])
#         self.base_model = m
#         self.head = Head(1000,1)
#         convert_sigmoid_to_mish(self.base_model)
#         convert_relu_to_mish(self.base_model)
        
        
    def forward(self, x):    
        #x= self.conv1(x)
        x = self.base(x)
        return x