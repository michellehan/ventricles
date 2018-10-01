import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights
import models as models

model_meta = {
        "resnet18":[8,6], "resnet34":[8,6], "resnet50":[8,6], "resnet101":[8,6], "resnet152":[8,6],
        #vgg16:[0,22], vgg19:[0,22],
        #resnext50:[8,6], resnext101:[8,6], resnext101_64:[8,6],
        #wrn:[8,6], inceptionresnet_2:[-2,9], inception_4:[-1,9],
        #dn121:[0,7], dn161:[0,7], dn169:[0,7], dn201:[0,7],
}



class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)
    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.decode(x)


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)





class UNet(nn.Module):
    #def __init__(self, num_classes):
    def __init__(self, num_classes, num_filters = 32, encoder = False):
        super(UNet, self).__init__()
 

        #self.encoder = models.__dict__[encoder](pretrained = True, num_classes = num_classes)

        ######
        #num_feat = [128, 256, 512, 1024, 2048]
        #mult = [1, 2, 4, 8, 16, 32]

        #self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        #self.enc1 = self.encoder.layer1 #256
        #self.enc2 = self.encoder.layer2 #512
        #self.enc3 = self.encoder.layer3 #1024
        #self.enc4 = self.encoder.layer4 #2048

        #self.center = DecoderBlock(2048, 4096, 1024)
        #self.dec4 = DecoderBlock(2048, 1024, 512)
        #self.dec3 = DecoderBlock(1024, 512, 256)
        #self.dec2 = DecoderBlock(512, 256, 128)
        #self.dec1 = DecoderBlock(256, 128, 64)
        #self.dec1 = DecoderBlock(128, 64, 32)
        #self.relu = ConvRelu(32, 32)


        #self.dec4 = DecoderBlock(2048 + num_filters * 8, 2048, num_filters * 8)
        #self.dec3 = DecoderBlock(1024 + num_filters * 8, 1024, num_filters * 8) 
        #self.dec2 = DecoderBlock(512 + num_filters * 8, 512, num_filters * 2)
        #self.dec1 = DecoderBlock(256 + num_filters * 2, num_filters * 4, num_filters * 2)
        #self.dec0 = DecoderBlock(num_filters * 2, num_filters, num_filters * 2)
        #self.relu = ConvRelu(num_filters * 2, num_filters)


        #self.final = nn.Conv2d(num_filters, 1, kernel_size = 1)

        #self.center = DecoderBlock(num_feat[4], num_filters * 8)
        #self.dec4 = DecoderBlock(num_feat[4] + num_filters * 8, num_filters * 8)
        #self.dec3 = DecoderBlock(num_feat[3] + num_filters * 8, num_filters * 8) 
        #self.dec2 = DecoderBlock(num_feat[2] + num_filters * 8, num_filters * 2)
        #self.dec1 = DecoderBlock(num_feat[1] + num_filters * 2, num_filters * 2 * 2)
        #self.dec0 = DecoderBlock(num_filters * 2 * 2, num_filters)
        #self.relu = ConvRelu(num_filters, num_filters)

        #self.final = nn.Conv2d(num_filters, 1, kernel_size = 1)

    #def forward(self, x):
        
        #enc0 = self.enc0(x)
        #enc1 = self.enc1(enc0)
        #enc2 = self.enc2(enc1)
        #enc3 = self.enc3(enc2)
        #enc4 = self.enc4(enc3)

        #center = self.center(nn.functional.max_pool2d(enc4, kernel_size = 2, stride = 1))

        #dec4 = self.dec4(torch.cat([enc4, center], dim = 1))
        #dec3 = self.dec3(torch.cat([enc3, dec4], dim = 1))
        #dec2 = self.dec2(torch.cat([enc2, dec3], dim = 1))
        #dec1 = self.dec1(torch.cat([enc1, dec2], dim = 1))
        #dec0 = self.dec0(torch.cat([enc0, dec1], dim=1))
        #dec0 = self.dec0(dec1)

        #relu = self.relu(dec0)
    
        #return self.final(relu)


        self.pool = nn.MaxPool2d(2,2)
        self.encoder = models.__dict__[encoder](pretrained = True).features
       

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)


    def forward(self, x):

        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)
