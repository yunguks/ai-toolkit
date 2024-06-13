import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, in_channel=1, in_dim=100, ngf=64):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.in_dim = in_dim
        self.in_channel = in_channel
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d(self.in_dim, self.ngf *4 , 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ngf*8) x 4 x 4``
            nn.ConvTranspose2d(self.ngf *4 , self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ngf*4) x 8 x 8``
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ngf*2) x 16 x 16``
            # nn.ConvTranspose2d( self.ngf, self.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf),
            # nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ngf) x 32 x 32``
            nn.ConvTranspose2d( self.ngf, self.in_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        # x = input
        # print("Generatoer")
        # for layer in self.main:
        #     print(f"x shape {x.shape} layer {type(layer)}")
        #     x = layer(x)
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, in_channel=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.in_channel = in_channel
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 ``(nc) x 32 x 32`` 입니다
            nn.Conv2d(in_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ndf) x 16 x 16``
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ndf*2) x 8 x 8``
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ndf*4) x 4 x 4``
            # nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ndf*8) x 4 x 4``
            nn.Conv2d(self.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # x = input
        # print("Discriminator")
        # for layer in self.main:
        #     print(f"x shape {x.shape} layer {type(layer)}")
        #     x = layer(x)
        return self.main(input)
    
    
class Encoder(nn.Module):
    def __init__(self,  letent_dim=2):
        super(Encoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(784,500),
            nn.LeakyReLU(0.01,inplace=True)
        )
        self.z_mean = nn.Linear(500,letent_dim)
        self.z_logvar = nn.Linear(500,letent_dim)
        
    def forward(self,x):
        x = self.layers(x)
        mean = self.z_mean(x)
        logvar = self.z_logvar(x)
        
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, letent_dim=2):
        super(Decoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(letent_dim,500),
            nn.LeakyReLU(0.01,inplace=True),
            nn.Linear(500,784),
        )
        
    def forward(self,x):
        # return self.layers(x)
        return torch.tanh(self.layers(x))
    
    
class VAEMLP(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAEMLP, self).__init__()
        
        self.encoder = encoder 
        self.decoder = decoder
        
    def reparameterization(self, mean, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std).cuda()
        return mean + std*eps
    
    def forward(self,x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        y = self.decoder(z)
        
        return y, mean, log_var
    
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(784, 500),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.fc2 = nn.Linear(500,10)
    
    def forward(self,x):
        x = self.layers(x)
        x = self.fc2(x)
        return x
        