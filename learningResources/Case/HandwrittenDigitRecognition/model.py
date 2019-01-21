# -*- encoding: utf-8
from torch import nn


# 使用方法：cnn = CNN(); target = argmax(cnn(input))
# 当使用对象作为函数时会调用nn.Module中的__call__()进而调用forword函数
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入为batch_size x 1 x 28 x 28
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3),
            nn.BatchNorm2d(num_features=25),  # 小批量归一化
            nn.ReLU(inplace=True)  # 对输入运用修正线性单元函数，覆盖运算
        )
        # 输出为batch_size x 25 x 26 x 26

        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化窗口大小为2，步长为2
        )
        # 输出为batch_size x 25 x 13 x 13

        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3),
            nn.BatchNorm2d(num_features=50),  # 小批量归一化
            nn.ReLU(inplace=True)  # 对输入运用修正线性单元函数，覆盖运算
        )
        # 输出为batch_size x 50 x 11 x 11

        self.layer_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 输出为batch_size x 50 x 5 x 5

        # 输入为batch_size x 50 * 5 * 5
        self.fc = nn.Sequential(
            nn.Linear(in_features=50 * 5 * 5, out_features=1024),
            # 输出为batch_size x 1024
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10)
        )
        # 输出为batch_size x 10

    def forward(self, x):
        # 将前一层的输出作为后一层的输入，顺次相连
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        # 将layer_4的输出展开成一维作为全连接层的输入
        x = x.view(x.size(0), -1)  # x.size(0)为batch_size
        x = self.fc(x)
        return x

