# -*- encoding: utf-8
from HandwrittenDigitRecognition.data_analysis import load_data
from HandwrittenDigitRecognition.model import CNN
from torch import nn, optim, from_numpy, max, save
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

EPOCH_NUM = 1  # 迭代次数
LEARNING_RATE = 0.02  # 学习率
BATCH_SIZE = 64  # 每批数量

# 加载数据
data_set = load_data()
train_dataset = TensorDataset(  # TensorDataset是torch.utils.data.Dataset的子类
    from_numpy(data_set[0]), from_numpy(data_set[1]))  # numpy数组转Tensor，特征和标签
train_loader = DataLoader(  # dataset参数需是torch.utils.data.Dataset的子类
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型
cnn = CNN()
# print(cnn)  # 输出cnn的结构信息

# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵最大损失函数

# 定义优化器
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE)  # 随机梯度下降优化器

for epoch in range(EPOCH_NUM):
    for i, (images, labels) in enumerate(train_loader):  # 对于每一个批次
        images = images.unsqueeze(1)  # batch_size x 28 x 28 -> batch_size x 1 x 28 x 28
        images = Variable(images)  # 模型使用Variable类型参数
        labels = Variable(labels)
        out = cnn(images)  # 获得模型预测结果
        loss = criterion(out, labels)  # 计算损失
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 优化器迭代一次
        if i % 10 == 9:
            print("Batch %d is over" % i)
save(cnn, 'cnn.pkl')  # 将模型整体保存到文件
