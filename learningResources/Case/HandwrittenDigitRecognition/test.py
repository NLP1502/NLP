# -*- encoding: utf-8
from HandwrittenDigitRecognition.data_analysis import load_data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.autograd import Variable
from torch import load, from_numpy, max, argmax
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

BATCH_SIZE = 64  # 每批数量
data_set = load_data()  # 获取测试集数据
test_dataset = TensorDataset(  # TensorDataset是torch.utils.data.Dataset的子类
    from_numpy(data_set[2]), from_numpy(data_set[3]))
test_loader = DataLoader(  # dataset参数需是torch.utils.data.Dataset的子类
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 从文件加载模型
cnn = load('cnn.pkl')

# 定义损失函数
criterion = CrossEntropyLoss()  # 交叉熵最大损失函数

# 将模型设置成evaluation模式，会影响BatchNorm
cnn.eval()

eval_loss = 0
eval_acc = 0
for images, labels in test_loader:
    images = images.unsqueeze(1)  # batch_size x 28 x 28 -> batch_size x 1 x 28 x 28
    images = Variable(images)  # 模型使用Variable类型参数
    out = cnn(images)  # 获得模型预测结果
    loss = criterion(out, labels)  # 计算损失
    eval_loss += loss.data.item() * labels.size(0)  # 损失率乘以每批数量得到损失数量
    _, pred = max(out, 1)  # 求维度1的argmax，batch_size x 10 -> batch_size x 1
    num_correct = (pred == labels).sum()  # 统计当前预测正确的标签数
    eval_acc += num_correct.item()  # 累计预测正确的标签数
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),  # 计算损失率
    eval_acc / (len(test_dataset))  # 计算正确率
))

# 肉眼观察预测的效果
for images, labels in test_loader:
    images = images.unsqueeze(1)
    images = Variable(images)
    out = cnn(images)
    for image, label in zip(images, out):
        print(argmax(label))  # 输出图片对应的数字
        plt.imshow(image.squeeze().numpy(), cmap='gray')  # 绘制手写数字的图片
        plt.show()

