import torch
import torch.nn as nn

"""Discriminator model arquitectures for images of 28 or 64 pixels

    Attributes:
        features_d	The lenght of the input image
        channels	The number of chanels of the input image, 1 -> gray scale
        labels		Total of diferent labels found at the dataset
    """
    
class DiscriminatorMannager():
    def __init__(self, img_size = 28, channels = 1, labels = 10, isDebugModeug = False):
        super(DiscriminatorMannager, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.labels = labels
        self.isDebugMode = isDebugModeug

    def get(self):
        if self.img_size == 28:
            print('#### Discriminator for 28x28 ####')
            return Discriminator28(self.img_size, self.channels, self.labels , self.isDebugMode)
        if self.img_size == 64:
            print('#### Discriminator for 64x64 ####')
            return Discriminator64(self.img_size, self.channels, self.labels, self.isDebugMode)


class Discriminator28(nn.Module):
    def __init__(self, img_size = 28, channels = 1, labels = 10, isDebugModeug = False):
        super(Discriminator28, self).__init__()
        self.isDebugMode = isDebugModeug

        self.layer_x = _input(channels, img_size, kernel_size=4, stride=2, padding=1)        # input: (1, 28, 28) 	=> 	out: (32, 14, 14)
        self.layer_y = _input(labels, img_size, kernel_size=4, stride=2, padding=1)	       # input: (10, 28, 28) 	=> 	out: (32, 14, 14)
        
        self.layer_1 = _block(img_size*2, img_size*4, kernel_size=4, stride=2, padding=1)  # input: (64, 14, 14)	=> 	out: (128, 7, 7) 
        self.layer_2 = _block(img_size*4, img_size*8, kernel_size=3, stride=2, padding=0)  # input: (128, 7, 7) 	=> 	out: (256, 3, 3)
        self.layer_3 = _output(img_size*8, channels, kernel_size=3, stride=1, padding=0)     # input: (256, 3, 3) 	=> 	out: (1, 1, 1)

    def forward(self, img, label):
        lx, ly = self.layer_x(img), self.layer_y(label)
        
        l0 = torch.cat([lx,ly], dim=1)
        if self.isDebugMode: print(f"_l0 = {l0.shape}") 
        
        l1 = self.layer_1(l0) 
        if self.isDebugMode: print(f"_l1 = {l1.shape}") 
        
        l2 = self.layer_2(l1)
        if self.isDebugMode: print(f"_l2 = {l2.shape}")
        
        l3 = self.layer_3(l2)
        if self.isDebugMode: print(f"_l3 = {l3.shape}")

        return l3.view(l3.shape[0], -1)
        
    def setMode(self, val):
        if val == 'debug':
            self.isDebugMode = True


class Discriminator64(nn.Module):
    def __init__(self, img_size = 64, channels = 1, labels = 10, isDebugModeug = False):
        super(Discriminator64, self).__init__()
        self.isDebugMode = isDebugModeug

        self.layer_x = _input(channels, img_size, kernel_size=4, stride=2, padding=1)        
        self.layer_y = _input(labels, img_size, kernel_size=4, stride=2, padding=1)	        
        
        self.layer_1 = _block(img_size*2, img_size*4, kernel_size=3, stride=2, padding=1)  
        self.layer_2 = _block(img_size*4, img_size*8, kernel_size=3, stride=2, padding=1)
        self.layer_3 = _block(img_size*8, img_size*16, kernel_size=3, stride=2, padding=1)
        self.layer_4 = _output(img_size*16, channels, kernel_size=4, stride=1, padding=0)     

    def forward(self, x, y):
        if self.isDebugMode: print(f"------DIS------")
        if self.isDebugMode: print(f"input = {x.shape} - {y.shape}")

        lx, ly = self.layer_x(x), self.layer_y(y)
        l0 = torch.cat([lx,ly], dim=1)
        if self.isDebugMode: print(f"_l0 = {l0.shape}") 

        l1 = self.layer_1(l0)
        if self.isDebugMode: print(f"_l1 = {l1.shape}") 
        
        l2 = self.layer_2(l1)
        if self.isDebugMode: print(f"_l2 = {l2.shape}")

        l3 = self.layer_3(l2)
        if self.isDebugMode: print(f"_l3 = {l3.shape}")
        
        l4 = self.layer_4(l3)
        if self.isDebugMode: print(f"_l4 = {l4.shape}")
        if self.isDebugMode: print(f"----------------")

        return l4.view(l4.shape[0], -1)
        

def _input(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            bias=False),
        nn.LeakyReLU(0.2, inplace=True)
)

def _block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def _output(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            bias=False), 
        nn.Sigmoid()
    )
