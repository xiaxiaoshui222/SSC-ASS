import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import models
import sys
arg1 = sys.argv[1]
db = arg1
data_type = 'chonggou'
save_path = 'savemodels/'+ data_type +'/' + db

if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = save_path + '/'
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 训练并保存模型参数
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

# 定义频谱聚类数据集
class MyDataSet(Dataset):
    def __init__(self, datasets,labels):
        self.len = datasets.shape[0]
        self.x_data = torch.FloatTensor(datasets)
        self.y_data = torch.from_numpy(labels)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len


def load_data(BATCH_SIZE=256, unlabel_size=0.7):

    # 加载频谱数据  第一次加载对数据进行划分train是含有标签  test是不含标签
    # 之后加载就直接使用
    train_data = np.load('train_sample/'+ data_type +'/'+ db +'/train_data.npy').reshape(-1, 1 ,64 ,64)
    test_data = np.load('train_sample/'+ data_type +'/'+ db +'/test_data.npy').reshape(-1, 1 ,64 ,64)
    train_label = np.load('train_sample/'+ data_type +'/'+ db +'/train_label.npy')
    test_label = np.load('train_sample/'+ data_type + '/'+ db +'/test_label.npy')

    print('labeled:', len(train_data))
    print('unlabeled:', len(test_data))

    trainSet = MyDataSet(train_data, train_label)
    trainloader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
    testSet = MyDataSet(test_data, test_label)
    testloader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True)

    return trainloader, testloader

def main():
    # 超参数设置
    EPOCH = 50  # 遍历数据集次数
    BATCH_SIZE = 256  # 批处理尺寸(batch_size)
    LR = 0.001  # 学习率
    num_clus = 3  # CNN分类器分类数量
    unlabel_size = 0.3  # 无标签数据占比

    model_save_path = './parameter64.pth'

    # 加载频谱数据
    trainloader, testloader = load_data(BATCH_SIZE, unlabel_size)
    print('premodel data size:', trainloader.dataset.x_data.shape)

    # 训练模型
    net = models.alexnet(out=num_clus).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, net.parameters()),  # 产生迭代器，x.requires_grad为真带入parameters()
        lr=LR,
    )

    # print(net)
    train(net, optimizer, criterion, EPOCH, trainloader, testloader, model_save_path)

if __name__ == "__main__":
    main()
    # mytest()
