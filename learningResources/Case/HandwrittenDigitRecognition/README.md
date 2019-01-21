# Hand-written Digit Recognition
## Introduce
使用卷积神经网络搭建的手写数字识别模型，开发环境:  
PyTorch=1.0, Python=3.5, OS=Windows10
## How to run
1. python train.py
2. python test.py 
## Model
五层卷积神经网络模型:
+ layer_1: 卷积层 + 小批量归一化 + ReLU[1 x 28 x 28 >> 25 x 26 x 26]  
+ layer_2: 池化层[25 x 26 x 26 >> 25 x 13 x 13]  
+ layer_3: 卷积层 + 小批量归一化 + ReLU[25 x 13 x 13 >> 50 x 11 x 11]  
+ layer_4: 池化层[50 x 11 x 11 >> 50 x 5 x 5]  
+ layer_fc: 全连接[50 * 5 * 5 >> 1024] + 全连接[1024 >> 10]  
## Files
./data_analysis.py: 解析数据，返回训练集和验证集的numpy数组  
./model.py: 模型的定义  
./train.py: 使用训练集训练模型并保存模型  
./test.py: 验证模型，计算损失率与正确率，输出图片和对应的标签以供观察  
./cnn.pkl: 保存模型的文件  
./MNIST/: 原始数据集，来自http://yann.lecun.com/exdb/mnist/?tdsourcetag=s_pcqq_aiomsg
## References
PyTorch基础入门六：PyTorch搭建卷积神经网络实现MNIST手写数字识别, https://blog.csdn.net/out_of_memory_error/article/details/81434907  
PyTorch中文文档, https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/