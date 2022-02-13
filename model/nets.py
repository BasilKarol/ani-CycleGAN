import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            )
        
        self.res_block_sample = [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(256)
        ]
        self.Res_block1 = nn.Sequential(*self.res_block_sample)
        self.Res_block2 = nn.Sequential(*self.res_block_sample)
        self.Res_block3 = nn.Sequential(*self.res_block_sample)
        self.Res_block4 = nn.Sequential(*self.res_block_sample)
        self.Res_block5 = nn.Sequential(*self.res_block_sample)
        self.Res_block6 = nn.Sequential(*self.res_block_sample)
        self.Res_block7 = nn.Sequential(*self.res_block_sample)
        self.Res_block8 = nn.Sequential(*self.res_block_sample)
        self.Res_block9 = nn.Sequential(*self.res_block_sample)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.downsample(x)
        
        
        #9 times for 256*256, 6 times for 128*128
        x = x + self.Res_block1(x)
        x = x + self.Res_block2(x)
        x = x + self.Res_block3(x)
        x = x + self.Res_block4(x)
        x = x + self.Res_block5(x)
        x = x + self.Res_block6(x)
        x = x + self.Res_block7(x)
        x = x + self.Res_block8(x)
        x = x + self.Res_block9(x)
        
        x = self.upsample(x)
        return x
    