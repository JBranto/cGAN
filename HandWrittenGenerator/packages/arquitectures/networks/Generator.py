import torch
import torch.nn as nn

"""Generator model arquitecture for images: 28px or 64px

    Attributes: 
        latent_size:     Latent space size, Vector Z
        labels_count:    Total of diferent labels_count found at the dataset
        img_size:        Image size
"""
class GeneratorMannager():
    def __init__(self, latent_size = 100, labels_count = 10, img_size = 28, isDebugMode = False ):
        super(GeneratorMannager, self).__init__()
        self.latent_size = latent_size
        self.labels_count = labels_count
        self.img_size = img_size
        self.isDebugMode = isDebugMode

    def get(self):
        if self.img_size == 28:
            print('#### Generator for 28x28 ####')
            return Generator28(self.latent_size, self.labels_count, self.isDebugMode)
        if self.img_size == 64:
            print('#### Generator for 64x64 ####')
            return Generator64(self.latent_size, self.labels_count, self.isDebugMode)
        

################################################################################################
class Generator28(nn.Module):
    def __init__(self, latent_size = 100, labels_count = 10, isDebugMode = False):
        super(Generator28, self).__init__()
        self.isDebugMode = isDebugMode

        self.layer_x = _input(latent_size, 128,kernel_size=3, stride=1, padding=0)	    # input: (100, 1, 1)	=>	out: (128, 3, 3)
        self.layer_y = _input(labels_count, 128, kernel_size=3, stride=1, padding=0)	# input: (10, 1, 1) 	=>	out: (128, 3, 3)

        self.layer_1 = _block(256, 128, kernel_size=3, stride=2, padding=0)		    # input: (256, 3, 3)	=>	out: (128, 7, 7)
        self.layer_2 = _block(128, 64, kernel_size=4, stride=2, padding=1)			# input: (128, 7, 7)	=>	out: (64, 14, 14)
        self.layer_3 = _output(64, 1, kernel_size=4, stride=2, padding=1)			# input: (64, 14, 14)	=>	out: (1, 28, 28)
        
    def forward(self, x, y):
        x, y = x.view(x.shape[0], x.shape[1], 1, 1), y.view(y.shape[0], y.shape[1], 1, 1)
        lx, ly = self.layer_x(x), self.layer_y(y)
        
        l0 = torch.cat([lx,ly], dim=1)
        if self.isDebugMode: print(f"_l0 = {l0.shape}")
    
        l1 = self.layer_1(l0) 
        if self.isDebugMode: print(f"_l1 = {l1.shape}") 
        
        l2 = self.layer_2(l1)
        if self.isDebugMode: print(f"_l2 = {l2.shape}")
        
        l3 = self.layer_3(l2)
        if self.isDebugMode: print(f"_l3 = {l3.shape}")

        return l3

################################################################################################
class Generator64(nn.Module):
    def __init__(self, latent_size=100, labels_count=10, isDebugMode = False):
        super(Generator64, self).__init__()
        self.isDebugMode = isDebugMode
        
        self.layer_x = _input(latent_size, 256,kernel_size=5, stride=1, padding=0) 
        self.layer_y = _input(labels_count, 256, kernel_size=5, stride=1, padding=0)    

        self.layer_1 = _block(512, 256, kernel_size=2, stride=2, padding=1)    
        self.layer_2 = _block(256, 128, kernel_size=4, stride=2, padding=1)   
        self.layer_3 = _block(128, 64, kernel_size=4, stride=2, padding=1)
        self.layer_4 = _output(64, 1, kernel_size=4, stride=2, padding=1)    

    def forward(self, x, y):
        if self.isDebugMode: print(f"------GEN------")
        if self.isDebugMode: print(f"Input = {x.shape} - {y.shape}")
        x, y = x.view(x.shape[0], x.shape[1], 1, 1), y.view(y.shape[0], y.shape[1], 1, 1)
        
        if self.isDebugMode: print(f"reshape = {x.shape} - {y.shape}")
        lx, ly = self.layer_x(x), self.layer_y(y)
    
        l0 = torch.cat([lx, ly], dim=1)
        if self.isDebugMode: print(f"_l0={l0.shape}")

        l1 = self.layer_1(l0)  
        if self.isDebugMode: print(f"_l1={l1.shape}")

        l2 = self.layer_2(l1)
        if self.isDebugMode: print(f"_l2={l2.shape}")

        l3 = self.layer_3(l2)
        if self.isDebugMode: print(f"_l3={l3.shape}")

        l4 = self.layer_4(l3)
        if self.isDebugMode: print(f"_l4={l4.shape}")
        if self.isDebugMode: print(f"----------------")

        return l4
    

# Convolutional Blocks
def _input(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def _block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def _output(in_channels, out_channels,kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
        nn.Tanh()	
    )
    