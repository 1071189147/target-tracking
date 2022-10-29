import torch.nn as nn
import torch.nn.functional as F
class ECAAttention(nn.Module):

    def __init__(self, num_channel,kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(num_channel,num_channel,groups=num_channel,kernel_size=(1,kernel_size),padding=(0,kernel_size//2))

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.reshape(b, c, h * w)
        x = x.reshape(b,h*w,1,c)
        x = self.conv(x)
        x = x.squeeze().permute(1,0).reshape(1,c,h,w)
        return x
class FPN(nn.Module):
    '''only for resnet50,101,152'''

    def __init__(self, features=256):
        super(FPN, self).__init__()
        self.reduce1 = nn.Conv2d(768, features, kernel_size=1)
        self.reduce2 = nn.Conv2d(384, features, kernel_size=1)
        self.reduce3 = nn.Conv2d(192, features, kernel_size=1)
        self.reduce4 = nn.Conv2d(96, features, kernel_size=1)
        self.s1 = nn.Sequential(
        nn.Sigmoid()
        )
        self.s2 = nn.Sequential(
        nn.Sigmoid()
        )
        self.s3 = nn.Sequential(
        nn.Sigmoid()
        )
        self.eca1 = ECAAttention(576)
        self.eca2 = ECAAttention(2304)
        self.eca3 = ECAAttention(9216)
        self.after_dsa_conv5 = nn.Conv2d(features,features,kernel_size=3,padding=1)
        self.after_dsa_conv4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.after_dsa_conv3 = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                             mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C2, C3, C4, C5 = x
        P5 = self.reduce1(C5)
        P4 = self.reduce2(C4)
        P3 = self.reduce3(C3)
        P2 = self.reduce4(C2)
        tmp = self.eca1(P4)
        tmp1 = self.upsamplelike([P5, C4])
        attentionmap = self.s1(tmp)
        P4 = tmp1 * attentionmap + P4
        P4 = self.after_dsa_conv5(P4)

        tmp = self.eca2(P3)
        tmp1 = self.upsamplelike([P4, C3])
        attentionmap = self.s2(tmp)
        P3 = tmp1 * attentionmap + P3
        P3 = self.after_dsa_conv4(P3)

        tmp = self.eca3(P2)
        tmp1 = self.upsamplelike([P3, C2])
        attentionmap = self.s3(tmp)
        P2 = tmp1 * attentionmap + P2
        P2 = self.after_dsa_conv3(P2)

        return [P2, P3, P4]


