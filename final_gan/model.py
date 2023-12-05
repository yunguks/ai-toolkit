import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channel=1, in_dim=100, ngf=64):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.in_dim = in_dim
        self.in_channel = in_channel
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d(self.in_dim, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(self.ngf*8) x 4 x 4``
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
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
    def __init__(self, in_channel=3, ndf=64):
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