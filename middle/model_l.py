import torch.nn as nn
import torchvision
import torch

class conv(nn.Module):
    def __init__(self,in_c,out_c,padding=1,dropout=0):
        super(conv,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        if dropout:
            self.conv.append(nn.Dropout(dropout))

    def forward(self,x):
        return self.conv(x)
    

class Block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0, repeat=1):
        super(Block, self).__init__()

        layers = []
        for _ in range(repeat):
            layers.append(conv(in_c,out_c,dropout = dropout))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):

        return self.layers(x)
    

class MyModel(nn.Module):
    def __init__(self, drop_out=0.5):
        super(MyModel, self).__init__()
        self.block1 = nn.Sequential(
            # 28 * 28
            conv(1, 16),
            Block(16, 16, drop_out,2),
            conv(16,32),
            Block(32, 32, drop_out,2),
            conv(32,64),
            Block(64, 64, drop_out,2),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            #14 * 14
            conv(64,128),
            Block(128, 128, drop_out,2),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            # 7 * 7
            Block(128, 128, drop_out,1),
            # 5 * 5
            conv(128,128,0),
            # 3 * 3
            conv(128,128,0),

        )
    
        
        self.classifier = nn.Sequential(
            nn.Linear(128,10),
        )


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x=   self.classifier(x)
        return x
