import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import itertools
import torchvision
import torchvision.transforms as transforms
import numpy as np
from IPython import display
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
import matplotlib.pyplot as plt

'''
class ResidualBlock(nn.Module):
    """Residual Block with batch or instance normalization."""
    def __init__(self, dim_in, dim_out,kernel_size,padding,stride=1,norm='Batch'):
        super(ResidualBlock, self).__init__()
        if (norm == 'Batch'):
            self.main = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=0, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim_out, dim_out, kernel_size, stride, padding=0, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))
        else:
            self.main = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=0, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim_out, dim_out, kernel_size, stride, padding=0, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
    def forward(self, x):
        return x + self.main(x)
'''

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size,padding):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ConvBlock(nn.Module):
    """Convolution Block with batch or instance normalization."""
    def __init__(self, dim_in, dim_out,kernel_size,padding,stride=1,norm='Batch'):
        super(ConvBlock, self).__init__()
        if (norm == 'Batch'):
            self.main = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=0, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True))
        else:
            self.main = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=0, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.main(x)

class ConvTBlock(nn.Module):
    """Transposed Convolution Block with batch or instance normalization."""
    def __init__(self, dim_in, dim_out,kernel_size,padding,stride=1,norm='Batch'):
        super(ConvTBlock, self).__init__()
        if (norm == 'Batch'):
            self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True))
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.main(x)

class AutoEncoder(nn.Module):
    def __init__(self,init_weights=True,norm='Batch',height = 128, width=416):
        super(AutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.downconv0 = nn.Conv2d(3,64,kernel_size=9,stride=1,padding=4,bias=False)
        self.downconv1 = nn.Conv2d(64,128,kernel_size=7,stride=2,padding=3,bias=False)
        self.downconv2 = nn.Conv2d(128,256,kernel_size=5,stride=2,padding=2,bias=False)
        self.downconv3 = nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,bias=False)
        ## down-sampling

        self.res64_down1  = ResidualBlock(64,64,9,4)
        self.res64_down2  = ResidualBlock(64,64,9,4)
        self.res64_up1  = ResidualBlock(64,64,9,4)
        self.res64_up2  = ResidualBlock(64,64,9,4)
        self.res128_down1 = ResidualBlock(128,128,7,3)
        self.res128_down2 = ResidualBlock(128,128,7,3)
        self.res128_up1 = ResidualBlock(128,128,7,3)
        self.res128_up2 = ResidualBlock(128,128,7,3)
        self.res256_down1 = ResidualBlock(256,256,5,2)
        self.res256_down2 = ResidualBlock(256,256,5,2)
        self.res256_up1 = ResidualBlock(256,256,5,2)
        self.res256_up2 = ResidualBlock(256,256,5,2)
        self.res512_1 = ResidualBlock(512,512,3,1)
        self.res512_2 = ResidualBlock(512,512,3,1)
        self.res512_3 = ResidualBlock(512,512,3,1)
        self.res512_4 = ResidualBlock(512,512,3,1)
        self.res512_5 = ResidualBlock(512,512,3,1)
        self.res512_6 = ResidualBlock(512,512,3,1)
        ## bottleneck block
        
        self.upconv0 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=1,padding=1,bias=False)
        self.upconv1 = nn.ConvTranspose2d(256,128,kernel_size=5,stride=1,padding=2,bias=False)
        self.upconv2 = nn.ConvTranspose2d(128,64,kernel_size=7,stride=1,padding=3,bias=False)
        self.upconv3 = nn.Conv2d(64,1,kernel_size=9,stride=1,padding=4,bias=False)
        ## up-sampling
        self.conv1x1_64 = nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv1x1_128 = nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv1x1_256 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=False)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if(norm == 'Batch'):
            print("- norm : Batch")
            self.N64_down = nn.BatchNorm2d(64, affine=True,track_running_stats=True)
            self.N128_down = nn.BatchNorm2d(128, affine=True,track_running_stats=True)
            self.N256_down = nn.BatchNorm2d(256, affine=True,track_running_stats=True)
            self.N512_down = nn.BatchNorm2d(512, affine=True,track_running_stats=True)
            self.N64_up = nn.BatchNorm2d(64, affine=True,track_running_stats=True)
            self.N128_up = nn.BatchNorm2d(128, affine=True,track_running_stats=True)
            self.N256_up = nn.BatchNorm2d(256, affine=True,track_running_stats=True)
        else:
            print("- norm : Instance")
            self.N64_down = nn.InstanceNorm2d(64, affine=True,track_running_stats=True)
            self.N128_down = nn.InstanceNorm2d(128, affine=True,track_running_stats=True)
            self.N256_down = nn.InstanceNorm2d(256, affine=True,track_running_stats=True)
            self.N512_down = nn.InstanceNorm2d(512, affine=True,track_running_stats=True)
            self.N64_up = nn.InstanceNorm2d(64, affine=True,track_running_stats=True)
            self.N128_up = nn.InstanceNorm2d(128, affine=True,track_running_stats=True)
            self.N256_up = nn.InstanceNorm2d(256, affine=True,track_running_stats=True)
            
        self.ReLU = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()
        

    def forward(self, x,istrain=True):
        x = x.cuda()
        # downconv0 block
        x1 = self.downconv0(x)
        x2 = self.N64_down(x1)
        x3 = self.ReLU(x2)

        #resblock64
        x4 = self.res64_down1(x3)
        x5 = self.res64_down2(x4)
        
        # downconv1 block
        x6 = self.downconv1(x5)
        x7 = self.N128_down(x6)
        x8 = self.ReLU(x7)

        #resblock128
        x9 = self.res128_down1(x8)
        x10 = self.res128_down2(x9)

        # downconv2 block
        x11 = self.downconv2(x10)
        x12 = self.N256_down(x11)
        x13 = self.ReLU(x12)

        #resblock256
        x14 = self.res256_down1(x13)
        x15 = self.res256_down2(x14)
        
        # downconv3 block
        x16 = self.downconv3(x15)
        x17 = self.N512_down(x16)
        x18 = self.ReLU(x17)
        
        # resblock512
        x18 = self.res512_1(x17)
        x19 = self.res512_2(x18)
        x20 = self.res512_3(x19)
        x21 = self.res512_4(x20)
        x22 = self.res512_5(x21)
        x23 = self.res512_6(x22)

        # upconv0 block
        x24 = self.upsampling(x23)
        x25 = self.upconv0(x24)
        x26 = self.N256_up(x25)
        x27 = self.ReLU(x26)

        #resblock256
        x27 = torch.cat((x27,x15),1)
        x27 = self.conv1x1_256(x27)
        x28 = self.res256_up1(x27)
        x29 = self.res256_up2(x28)
        
        # upconv1 block
        x30 = self.upsampling(x29)
        x31 = self.upconv1(x30)
        x32 = self.N128_up(x31)
        x33 = self.ReLU(x32)

        #resblock128
        x33 = torch.cat((x33,x10),1)
        x33 = self.conv1x1_128(x33)
        x34 = self.res128_up1(x33)
        x35 = self.res128_up2(x34)
       
        # upconv2 blcok
        x36 = self.upsampling(x35)
        x37 = self.upconv2(x36)   
        x38 = self.N64_up(x37)
        x39 = self.ReLU(x38)

        #resblock64
        x39 = torch.cat((x39,x5),1)
        x39 = self.conv1x1_64(x39)
        x40 = self.res64_up1(x39)
        x41 = self.res64_up2(x40)

        # upconv3 blcok
        x42 = self.upconv3(x41)   
        x43 = x42.tanh().clone()
        x44 = x43.view(-1,1,self.height,self.width)

        if istrain is True:
            return x5, x10, x15, x23, x29,x35,x41,x44
        else:
            return x44

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n*=k
                stdv = 1. /math.sqrt(n)
                m.weight.data.uniform_(-stdv,stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv,stdv)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()                
                
class AutoEncoder_2(nn.Module):
    def __init__(self,init_weights=True, norm='Batch',input_dim=3,height = 128, width=416):
        super(AutoEncoder_2, self).__init__()
        if(norm == 'Batch'):
            print("- norm : Batch")
        else:
            print("- norm : Instance")
        self.height = height
        self.width = width
        self.downconv0 = ConvBlock(input_dim,64,kernel_size=9,stride=1,padding=4)
        self.downconv1 = ConvBlock(64,128,kernel_size=7,stride=2,padding=3)
        self.downconv2 = ConvBlock(128,256,kernel_size=5,stride=2,padding=2)
        self.downconv3 = ConvBlock(256,512,kernel_size=3,stride=2,padding=1)
        self.downconv4 = ConvBlock(512,512,kernel_size=3,stride=2,padding=1)
        ## down-sampling
        self.res64_down1  = ResidualBlock(64,64,9,4)
        self.res64_up1  = ResidualBlock(64,64,9,4)
        self.res128_down1 = ResidualBlock(128,128,7,3)
        self.res128_up1 = ResidualBlock(128,128,7,3)
        self.res256_down1 = ResidualBlock(256,256,5,2)
        self.res256_up1 = ResidualBlock(256,256,5,2)
        self.res512_down1 = ResidualBlock(512,512,3,1)
        self.res512_up1 = ResidualBlock(512,512,3,1)
        self.res512_down2 = ResidualBlock(512,512,3,1)
        self.res512_up2 = ResidualBlock(512,512,3,1)
        self.res512_1 = ResidualBlock(512,512,3,1)
        self.res512_2 = ResidualBlock(512,512,3,1)
        self.res512_3 = ResidualBlock(512,512,3,1)
        self.res512_4 = ResidualBlock(512,512,3,1)
        self.res512_5 = ResidualBlock(512,512,3,1)
        self.res512_6 = ResidualBlock(512,512,3,1)
        ## bottleneck block        
        self.upconv0 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.upconv1 = ConvBlock(512,256,kernel_size=3,stride=1,padding=1)
        self.upconv2 = ConvBlock(256,128,kernel_size=5,stride=1,padding=2)
        self.upconv3 = ConvBlock(128,64,kernel_size=7,stride=1,padding=3)
        #self.pad4 = nn.ReflectionPad2d(4)
        self.upconv4 = nn.Conv2d(64,1,kernel_size=9,stride=1,padding=4,bias=False)
        ## up-sampling
        self.conv1x1_64 = ConvBlock(128,64,kernel_size=1,stride=1,padding=0)
        self.conv1x1_128 = ConvBlock(256,128,kernel_size=1,stride=1,padding=0)
        self.conv1x1_256 = ConvBlock(512,256,kernel_size=1,stride=1,padding=0)
        self.conv1x1_512 = ConvBlock(1024,512,kernel_size=1,stride=1,padding=0)
        #self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsampling = nn.functional.interpolate
            
        self.ReLU = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()
    def forward(self, x,istrain=False):
        # downconv0 block
        x1_cat = self.downconv0(x)              # 128 x 416, 64ch
        x1 = self.res64_down1(x1_cat)                 
        # downconv1 block                    
        x2_cat = self.downconv1(x1)             # 64 x 208, 128ch
        x2 = self.res128_down1(x2_cat)          
        # downconv2 block
        x3_cat = self.downconv2(x2)             # 32 x 104, 256ch
        x3 = self.res256_down1(x3_cat)          
        # downconv3 block
        x4_cat = self.downconv3(x3)             # 16 x 52, 512ch
        x4 = self.res512_down1(x4_cat)
        x4 = self.res512_down2(x4)           
        # downconv4 block
        x5 = self.downconv4(x4)                 # 8 x 26, 512ch
        # resblock512 - Bottleneck
        x6 = self.res512_1(x5)
        x6 = self.res512_2(x6)
        x6 = self.res512_3(x6)
        x6 = self.res512_4(x6)
        x6 = self.res512_5(x6)
        x6 = self.res512_6(x6)
        # upconv0 block
        x7 = self.upsampling(x6, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x7 = self.upconv0(x7)                   # 16 x 52, 512ch
        x8 = torch.cat((x7,x4_cat),1)           # 16 x 52, 1024ch
        x8 = self.conv1x1_512(x8)             
        x8 = self.res512_up1(x8) 
        x8 = self.res512_up2(x8)                # 16 x 52, 512ch (1024 -> 512)
        # upconv1 block
        x9 = self.upsampling(x8, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x9 = self.upconv1(x9)                   # 32 x 104, 256ch
        x10 = torch.cat((x9,x3_cat),1)          # 32 x 104, 512ch
        x10 = self.conv1x1_256(x10)
        x10 = self.res256_up1(x10)              # 32 x 104, 256ch (512 -> 256)
        # upconv2 block
        x11 = self.upsampling(x10, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x11 = self.upconv2(x11)                 # 64 x 208, 128ch
        x12 = torch.cat((x11,x2_cat),1)         # 64 x 208, 256ch
        x12 = self.conv1x1_128(x12)
        x12 = self.res128_up1(x12)              # 64 x 208, 128ch (256 -> 128)
        # upconv3 block
        x13 = self.upsampling(x12, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x13 = self.upconv3(x13)                 # 128 x 416, 64ch
        x14 = torch.cat((x13,x1_cat),1)         # 128 x 416, 128ch
        x14 = self.conv1x1_64(x14)
        x14 = self.res64_up1(x14)               # 128 x 416, 64ch (128 -> 64)
        # upconv5 block
        #x15 = self.pad4(x14)
        x15 = self.upconv4(x14)                 # 128 x 416, 1ch
        x15 = x15.tanh().clone()
        x15 = x15.view(-1,1,self.height,self.width)
        if istrain is True:
            return x1, x2, x4, x6, x8,x12,x14,x15
        else:
            return x15

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n*=k
                stdv = 1. /math.sqrt(n)
                m.weight.data.uniform_(-stdv,stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv,stdv)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class AutoEncoder_Unet(nn.Module):
    def __init__(self,init_weights=True, norm='Batch',input_dim=3,height = 128, width=416):
        super(AutoEncoder_Unet, self).__init__()
        if(norm == 'Batch'):
            print("- norm : Batch")
        else:
            print("- norm : Instance")
        self.height = height
        self.width = width
        self.downconv0 = ConvBlock(input_dim,64,kernel_size=9,stride=1,padding=4)
        self.downconv1 = ConvBlock(64,128,kernel_size=7,stride=2,padding=3)
        self.downconv2 = ConvBlock(128,256,kernel_size=5,stride=2,padding=2)
        self.downconv3 = ConvBlock(256,512,kernel_size=3,stride=2,padding=1)
        self.downconv4 = ConvBlock(512,512,kernel_size=3,stride=2,padding=1)
        ## down-sampling
        self.conv512_1 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.conv512_2 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.conv512_3 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.conv512_4 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.conv512_5 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.conv512_6 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)

        self.upconv0 = ConvBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.upconv1 = ConvBlock(512,256,kernel_size=3,stride=1,padding=1)
        self.upconv2 = ConvBlock(256,128,kernel_size=5,stride=1,padding=2)
        self.upconv3 = ConvBlock(128,64,kernel_size=7,stride=1,padding=3)
        self.upconv4 = nn.Conv2d(64,1,kernel_size=9,stride=1,padding=4,bias=False)
        ## up-sampling
        self.conv1x1_64 = ConvBlock(128,64,kernel_size=1,stride=1,padding=0)
        self.conv1x1_128 = ConvBlock(256,128,kernel_size=1,stride=1,padding=0)
        self.conv1x1_256 = ConvBlock(512,256,kernel_size=1,stride=1,padding=0)
        self.conv1x1_512 = ConvBlock(1024,512,kernel_size=1,stride=1,padding=0)
        #self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsampling = nn.functional.interpolate
            
        self.ReLU = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()
    def forward(self, x,istrain=False):
        # downconv0 block
        x1_cat = self.downconv0(x)              # 128 x 416, 64ch
        # downconv1 block                    
        x2_cat = self.downconv1(x1_cat)             # 64 x 208, 128ch
        # downconv2 block
        x3_cat = self.downconv2(x2_cat)             # 32 x 104, 256ch
        # downconv3 block
        x4_cat = self.downconv3(x3_cat)             # 16 x 52, 512ch
        # downconv4 block
        x5 = self.downconv4(x4_cat)                 # 8 x 26, 512ch

        x6 = self.conv512_1(x5)
        x6 = self.conv512_2(x6)
        x6 = self.conv512_3(x6)
        x6 = self.conv512_4(x6)
        x6 = self.conv512_5(x6)
        x6 = self.conv512_6(x6)

        # upconv0 block
        x7 = self.upsampling(x6, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x7 = self.upconv0(x7)                   # 16 x 52, 512ch
        x8 = torch.cat((x7,x4_cat),1)           # 16 x 52, 1024ch
        x8 = self.conv1x1_512(x8)             
        # upconv1 block
        x9 = self.upsampling(x8, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x9 = self.upconv1(x9)                   # 32 x 104, 256ch
        x10 = torch.cat((x9,x3_cat),1)          # 32 x 104, 512ch
        x10 = self.conv1x1_256(x10)
        # upconv2 block
        x11 = self.upsampling(x10, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x11 = self.upconv2(x11)                 # 64 x 208, 128ch
        x12 = torch.cat((x11,x2_cat),1)         # 64 x 208, 256ch
        x12 = self.conv1x1_128(x12)
        # upconv3 block
        x13 = self.upsampling(x12, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x13 = self.upconv3(x13)                 # 128 x 416, 64ch
        x14 = torch.cat((x13,x1_cat),1)         # 128 x 416, 128ch
        x14 = self.conv1x1_64(x14)
        # upconv5 block
        x15 = self.upconv4(x14)                 # 128 x 416, 1ch
        x15 = x15.tanh().clone()
        x15 = x15.view(-1,1,self.height,self.width)
        if istrain is True:
            return x1_cat, x2_cat, x4_cat, x6, x8,x12,x14,x15
        else:
            return x15

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n*=k
                stdv = 1. /math.sqrt(n)
                m.weight.data.uniform_(-stdv,stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv,stdv)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class AutoEncoder_DtoD(nn.Module):
    def __init__(self,init_weights=True, norm='Batch',input_dim=1,height = 128, width=416):
        super(AutoEncoder_DtoD, self).__init__()
        if(norm == 'Batch'):
            print("- norm : Batch")
        else:
            print("- norm : Instance")
        self.height = height
        self.width = width
        self.downconv0 = ConvBlock(input_dim,64,kernel_size=9,stride=1,padding=4)
        self.downconv1 = ConvBlock(64,128,kernel_size=4,stride=2,padding=1)
        self.downconv2 = ConvBlock(128,256,kernel_size=4,stride=2,padding=1)
        self.downconv3 = ConvBlock(256,512,kernel_size=4,stride=2,padding=1)
        self.downconv4 = ConvBlock(512,512,kernel_size=4,stride=2,padding=1)
        ## down-sampling
        self.res64_down1  = ResidualBlock(64,64,9,4)
        self.res64_up1  = ResidualBlock(64,64,9,4)
        self.res128_down1 = ResidualBlock(128,128,7,3)
        self.res128_up1 = ResidualBlock(128,128,7,3)
        self.res256_down1 = ResidualBlock(256,256,5,2)
        self.res256_up1 = ResidualBlock(256,256,5,2)
        self.res512_down1 = ResidualBlock(512,512,3,1)
        self.res512_up1 = ResidualBlock(512,512,3,1)
        self.res512_down2 = ResidualBlock(512,512,3,1)
        self.res512_up2 = ResidualBlock(512,512,3,1)
        self.res512_1 = ResidualBlock(512,512,3,1)
        self.res512_2 = ResidualBlock(512,512,3,1)
        self.res512_3 = ResidualBlock(512,512,3,1)
        self.res512_4 = ResidualBlock(512,512,3,1)
        self.res512_5 = ResidualBlock(512,512,3,1)
        self.res512_6 = ResidualBlock(512,512,3,1)
        ## bottleneck block        
        self.upconv0 = ConvTBlock(512,512,kernel_size=4,stride=2,padding=1)
        self.upconv1 = ConvTBlock(512,256,kernel_size=4,stride=2,padding=1)
        self.upconv2 = ConvTBlock(256,128,kernel_size=4,stride=2,padding=1)
        self.upconv3 = ConvTBlock(128,64,kernel_size=4,stride=2,padding=1)
        self.upconv4 = nn.ConvTranspose2d(64,1,kernel_size=9,stride=1,padding=4,bias=False)
        ## up-sampling
        #self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsampling = nn.functional.interpolate
            
        self.ReLU = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()
    def forward(self, x,istrain=False):
        # downconv0 block
        x1_cat = self.downconv0(x)              # 128 x 416, 64ch
        x1 = self.res64_down1(x1_cat)                 
        # downconv1 block                    
        x2_cat = self.downconv1(x1)             # 64 x 208, 128ch
        x2 = self.res128_down1(x2_cat)          
        # downconv2 block
        x3_cat = self.downconv2(x2)             # 32 x 104, 256ch
        x3 = self.res256_down1(x3_cat)          
        # downconv3 block
        x4_cat = self.downconv3(x3)             # 16 x 52, 512ch
        x4 = self.res512_down1(x4_cat)
        x4 = self.res512_down2(x4)           
        # downconv4 block
        x5 = self.downconv4(x4)                 # 8 x 26, 512ch
        # resblock512 - Bottleneck
        x6 = self.res512_1(x5)
        x6 = self.res512_2(x6)
        x6 = self.res512_3(x6)
        x6 = self.res512_4(x6)
        x6 = self.res512_5(x6)
        x6 = self.res512_6(x6)
        # upconv0 block
        #x7 = self.upsampling(x6, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x7 = self.upconv0(x6)                   # 16 x 52, 512ch
        x8 = self.res512_up1(x7) 
        x8 = self.res512_up2(x8)                # 16 x 52, 512ch
        # upconv1 block
        #x9 = self.upsampling(x8, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x9 = self.upconv1(x8)                   # 32 x 104, 256ch
        x10 = self.res256_up1(x9)              # 32 x 104, 256ch
        # upconv2 block
        #x11 = self.upsampling(x10, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x11 = self.upconv2(x10)                 # 64 x 208, 128ch
        x12 = self.res128_up1(x11)              # 64 x 208, 128ch
        # upconv3 block
        #x13 = self.upsampling(x12, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x13 = self.upconv3(x12)                 # 128 x 416, 64ch
        x14 = self.res64_up1(x13)               # 128 x 416, 64ch
        # upconv5 block
        x15 = self.upconv4(x14)                 # 128 x 416, 1ch
        x15 = x15.tanh().clone()
        x15 = x15.view(-1,1,self.height,self.width)
        if istrain is True:
            return x1, x2, x4, x6, x8,x12,x14,x15
        else:
            return x15

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n*=k
                stdv = 1. /math.sqrt(n)
                m.weight.data.uniform_(-stdv,stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv,stdv)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class AutoEncoder_Resnet(nn.Module):
    def __init__(self,init_weights=True, norm='Batch',input_dim=3,height = 128, width=416):
        super(AutoEncoder_Resnet, self).__init__()
        if(norm == 'Batch'):
            print("- norm : Batch")
        else:
            print("- norm : Instance")
        self.height = height
        self.width = width
        self.downconv0 = ConvBlock(input_dim,64,kernel_size=9,stride=1,padding=4)
        self.downconv1 = ConvBlock(64,128,kernel_size=7,stride=2,padding=3)
        self.downconv2 = ConvBlock(128,256,kernel_size=5,stride=2,padding=2)
        self.downconv3 = ConvBlock(256,512,kernel_size=3,stride=2,padding=1)
        self.downconv4 = ConvBlock(512,512,kernel_size=3,stride=2,padding=1)
        ## down-sampling
        self.res64_down1  = ResidualBlock(64,64,9,4)
        self.res64_up1  = ResidualBlock(64,64,9,4)
        self.res128_down1 = ResidualBlock(128,128,7,3)
        self.res128_up1 = ResidualBlock(128,128,7,3)
        self.res256_down1 = ResidualBlock(256,256,5,2)
        self.res256_up1 = ResidualBlock(256,256,5,2)
        self.res512_down1 = ResidualBlock(512,512,3,1)
        self.res512_up1 = ResidualBlock(512,512,3,1)
        self.res512_down2 = ResidualBlock(512,512,3,1)
        self.res512_up2 = ResidualBlock(512,512,3,1)
        self.res512_1 = ResidualBlock(512,512,3,1)
        self.res512_2 = ResidualBlock(512,512,3,1)
        self.res512_3 = ResidualBlock(512,512,3,1)
        self.res512_4 = ResidualBlock(512,512,3,1)
        self.res512_5 = ResidualBlock(512,512,3,1)
        self.res512_6 = ResidualBlock(512,512,3,1)
        ## bottleneck block        
        self.upconv0 = ConvTBlock(512,512,kernel_size=3,stride=1,padding=1)
        self.upconv1 = ConvTBlock(512,256,kernel_size=3,stride=1,padding=1)
        self.upconv2 = ConvTBlock(256,128,kernel_size=5,stride=1,padding=2)
        self.upconv3 = ConvTBlock(128,64,kernel_size=7,stride=1,padding=3)
        self.upconv4 = nn.Conv2d(64,1,kernel_size=9,stride=1,padding=4,bias=False)
        ## up-sampling
        #self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsampling = nn.functional.interpolate
            
        self.ReLU = nn.ReLU(inplace=True)
        if init_weights:
            self._initialize_weights()
    def forward(self, x,istrain=False):
        # downconv0 block
        x1_cat = self.downconv0(x)              # 128 x 416, 64ch
        x1 = self.res64_down1(x1_cat)                 
        # downconv1 block                    
        x2_cat = self.downconv1(x1)             # 64 x 208, 128ch
        x2 = self.res128_down1(x2_cat)          
        # downconv2 block
        x3_cat = self.downconv2(x2)             # 32 x 104, 256ch
        x3 = self.res256_down1(x3_cat)          
        # downconv3 block
        x4_cat = self.downconv3(x3)             # 16 x 52, 512ch
        x4 = self.res512_down1(x4_cat)
        x4 = self.res512_down2(x4)           
        # downconv4 block
        x5 = self.downconv4(x4)                 # 8 x 26, 512ch
        # resblock512 - Bottleneck
        x6 = self.res512_1(x5)
        x6 = self.res512_2(x6)
        x6 = self.res512_3(x6)
        x6 = self.res512_4(x6)
        x6 = self.res512_5(x6)
        x6 = self.res512_6(x6)
        # upconv0 block
        x7 = self.upsampling(x6, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x7 = self.upconv0(x7)                   # 16 x 52, 512ch
        x8 = self.res512_up1(x7) 
        x8 = self.res512_up2(x8)                # 16 x 52, 512ch
        # upconv1 block
        x9 = self.upsampling(x8, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x9 = self.upconv1(x9)                   # 32 x 104, 256ch
        x10 = self.res256_up1(x9)              # 32 x 104, 256ch
        # upconv2 block
        x11 = self.upsampling(x10, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x11 = self.upconv2(x11)                 # 64 x 208, 128ch
        x12 = self.res128_up1(x11)              # 64 x 208, 128ch
        # upconv3 block
        x13 = self.upsampling(x12, scale_factor = 2, mode = 'bilinear', align_corners=False)
        x13 = self.upconv3(x13)                 # 128 x 416, 64ch
        x14 = self.res64_up1(x13)               # 128 x 416, 64ch
        # upconv5 block
        x15 = self.upconv4(x14)                 # 128 x 416, 1ch
        x15 = x15.tanh().clone()
        x15 = x15.view(-1,1,self.height,self.width)
        if istrain is True:
            return x1, x2, x4, x6, x8,x12,x14,x15
        else:
            return x15

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n*=k
                stdv = 1. /math.sqrt(n)
                m.weight.data.uniform_(-stdv,stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv,stdv)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


        
        
