import torch
from sympy.solvers.diophantine.diophantine import Linear
from torch import nn
from torchsummary import summary

#Inception模块
class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.ReLU=nn.ReLU()

        #路线一，1*1卷积
        self.p1_1=nn.Conv2d(in_channels=in_channels,out_channels=c1,kernel_size=1)

        #路线二，1*1卷积，3*3卷积
        self.p2_1=nn.Conv2d(in_channels=in_channels,out_channels=c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,padding=1)

        #路线三，1*1卷积，5*5卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        #路线四，3*3最大池化，1*1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)        #最大池化的默认步幅等于卷积核大小
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4,kernel_size=1)

    #Inception模块的前向传播。四条路线合并
    def forward(self,x):
        p1=self.ReLU(self.p1_1(x))
        p2=self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3=self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4=self.ReLU(self.p4_2(self.p4_1(x)))
        #dim=1表示按通道融合
        return torch.cat((p1,p2,p3,p4),dim=1)

#GoogLeNet网络搭建,辅助分类器(过个几层网络显示分类结果已更新前面网络的loss值)不写
class GoogLeNet(nn.Module):
    def __init__(self,Inception):
        super(GoogLeNet,self).__init__()

        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        #Inception块
        self.b3=nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),         #1
            Inception(256,128,(128,192),(32,96),64),       #2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),    #3
            Inception(512, 160, (112, 224), (24, 64), 64),   #4
            Inception(512, 128, (128, 256), (24, 64), 64),   #5
            Inception(512, 112, (128, 288), (32, 64), 64),   #6
            Inception(528, 256, (160, 320), (32, 128), 128), #7
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5=nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),  #8
            Inception(832, 384, (192, 384), (48, 128), 128),  #9
            #全局平均池化（将输入的832*832大小的图像用1*1的代替）
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,10)
        )

        # 由于网络较深，出现梯度消失，随机w和b可能离真实值差太大，无法收敛，要进行权重初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        return x

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=GoogLeNet(Inception).to(device)
    #summary函数输出网络参数
    print(summary(model,(1,224,224)))






