import os
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from semilearn.core.utils import get_logger

# import sys
# arg1 = sys.argv[1]
db = '-24'
data_type = 'hecheng'
save_path = 'baseline_savemodels/'+ data_type +'/' +db

if not os.path.exists(save_path):
    os.makedirs(save_path)
logger = get_logger('cwq',save_path=save_path, level="INFO")
save_path = save_path + '/'

print(logger)

print_fn = print if logger is None else logger.info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_fn('device:{}'.format(device))
np.random.seed(1234)
#卷积模块，由卷积核和激活函数组成
class conv_block(nn.Module):
    def __init__(self,ks,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=ks,stride=1,padding=1,bias=True),  #二维卷积核，用于提取局部的图像信息
            nn.ReLU(inplace=True), #这里用ReLU作为激活函数
            nn.Conv2d(ch_out, ch_out, kernel_size=ks,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)


# 常规CNN模块（由几个卷积模块堆叠而成）
class CNN(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(CNN, self).__init__()
        feature_list = [16, 32, 64, 128, 256]  # 代表每一层网络的特征数，扩大特征空间有助于挖掘更多的局部信息
        self.conv1 = conv_block(kernel_size, in_ch, feature_list[0])
        self.conv2 = conv_block(kernel_size, feature_list[0], feature_list[1])
        self.conv3 = conv_block(kernel_size, feature_list[1], feature_list[2])
        self.conv4 = conv_block(kernel_size, feature_list[2], feature_list[3])
        self.conv5 = conv_block(kernel_size, feature_list[3], feature_list[4])
        self.fc = nn.Sequential(  # 全连接层主要用来进行分类，整合采集的局部信息以及全局信息
            nn.Linear(feature_list[4] * 64 * 64, 1024),  # 此处28为MINST一张图片的维度
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        device = x.device
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.view(x5.size()[0], -1)  # 全连接层相当于做了矩阵乘法，所以这里需要将维度降维来实现矩阵的运算
        out = self.fc(x5)
        return out

def load_data(BATCH_SIZE=256):
    # 加载 在getPreModel.py 中已经划分好了的频谱数据
    train_data = np.load('train_sample/'+ data_type +'/'+ db +'/train_data.npy').reshape(-1, 1 ,64 ,64)
    test_data = np.load('train_sample/'+ data_type +'/'+ db +'/test_data.npy').reshape(-1, 1 ,64 ,64)
    train_label = np.load('train_sample/'+ data_type +'/'+ db +'/train_label.npy')
    test_label = np.load('train_sample/'+ data_type + '/'+ db +'/test_label.npy')

    # 加载全部数据  将train当做有标签  test当做无标签
    dataset = np.concatenate((train_data, test_data), axis=0)
    label = np.concatenate((train_label, test_label), axis=0)
    train_data, test_data, train_label, test_label = train_test_split(dataset, label, test_size=27000)
    trainset = MyDataSet(train_data, train_label)
    trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = MyDataSet(test_data, test_label)
    testloader = Data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    print_fn('labeld:{}'.format(trainset.len))
    print_fn('unlabeled:{}'.format(testset.len))
    return trainloader, testloader


# 定义频谱聚类数据集
class MyDataSet(Data.Dataset):
    def __init__(self, datasets,labels):
        self.len = datasets.shape[0]
        self.x_data = torch.FloatTensor(datasets)
        self.y_data = torch.from_numpy(labels)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

def train(net, optimizer, criterion, EPOCH, trainloader, testloader, model_save_path):
    bast_model = None
    bast_acc = 0

    for epoch in range(EPOCH):
        net.train()
        sum_loss = 0.0

        # if bast_acc!=0:
        #     net.load_state_dict(bast_model)

        true_label = []
        pred_label = []
        for i, data in enumerate(trainloader):
            inputs, labels = data
            true_label += list(labels.numpy())
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            # 获取train上的预测结果
            _, predicted = torch.max(outputs.data, 1)
            pred_label += list(predicted.cpu().numpy())

            # backward 更新参数
            loss.backward()
            optimizer.step()

            # 每训练10个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 10))
                sum_loss = 0.0
        # train上的准确度
        acc_train = accuracy_score(true_label, pred_label)*100

        # 每跑完一次epoch测试一下在test上的准确率
        net.eval()
        with torch.no_grad():
            correct = 0.
            total = 0.
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum()
            acc = (100 * correct / total)
            print('第%d个epoch，train上识别准确率为：%.3f%%, test上识别准确率为：%.3f%%'
                  % (epoch + 1,acc_train, acc))

            if acc > bast_acc: # 保存泛化能力最好的模型
                bast_acc = acc
                bast_model = net.state_dict()

    # 保存模型参数
    torch.save(bast_model, model_save_path)
    print('model is saved to {}'.format(model_save_path))

def main():

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-8)  # 使用Adam优化器
    # criterion = nn.CrossEntropyLoss()  # 分类任务常用的交叉熵损失函数
    # 超参数设置
    EPOCH = 200  # 遍历数据集次数
    BATCH_SIZE = 64  # 批处理尺寸(batch_size)
    LR = 0.001  # 学习率
    num_clus = 3  # CNN分类器分类数量

    model_save_path = save_path + 'parameter64.pth'

    # 加载频谱数据
    trainloader, testloader = load_data(BATCH_SIZE)
    print('premodel data size:', trainloader.dataset.x_data.shape)

    # 训练模型
    net = CNN(3, 1, 1).to(device=device, dtype=torch.float32)
    # net = models.alexnet(out=num_clus).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, net.parameters()),  # 产生迭代器，x.requires_grad为真带入parameters()
        lr=LR,
    )

    # print(net)
    train(net, optimizer, criterion, EPOCH, trainloader, testloader, model_save_path)
if __name__ == '__main__':
    main()
