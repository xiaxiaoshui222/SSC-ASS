import os
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from semilearn.core.utils import get_logger

import models

import sys
arg1 = sys.argv[1]
db = arg1
data_type = 'zhenshi'
save_path = 'baseline_savemodels/'+ data_type +'/' +db

if not os.path.exists(save_path):
    os.makedirs(save_path)
logger = get_logger('cwq',save_path=save_path, level="INFO")
save_path = save_path + '/'

print(logger)

print_fn = print if logger is None else logger.info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_fn('device:{}'.format(device))

def load_data(BATCH_SIZE=256):
    # 加载 在getPreModel.py 中已经划分好了的频谱数据
    train_data = np.load('train_sample/'+ data_type +'/'+ db +'/train_data.npy').reshape(-1, 1 ,128 ,128)
    test_data = np.load('train_sample/'+ data_type +'/'+ db +'/test_data.npy').reshape(-1, 1 ,128 ,128)
    train_label = np.load('train_sample/'+ data_type +'/'+ db +'/train_label.npy')
    test_label = np.load('train_sample/'+ data_type + '/'+ db +'/test_label.npy')

    # 加载全部数据  将train当做有标签  test当做无标签
    dataset = np.concatenate((train_data, test_data), axis=0)
    label = np.concatenate((train_label, test_label), axis=0)
    train_data, test_data, train_label, test_label = train_test_split(dataset, label, test_size=5500)
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

#
# def main():
#     nums_clu = 3
#     BATCH_SIZE = 256
#     EPOCH = 200
#
#     # load data
#     trainloader, testloader, premodel, finalmodel = load_data()
#
#
#     finalmodel = save_path + finalmodel
#     # CNN
#     model = models.alexnet(out=nums_clu)
#     # print_fn(model)
#     model.to(device)
#     # # 加载预训练模型=======================================================
#     # print_fn('load pre-trained model form:', premodel)
#     # para = torch.load(premodel)
#     # model.load_state_dict(para)
#     acc = cnn_acc(model, testloader)
#     print_fn('pre-trained model on unlabeled acc:{}'.format(acc))
#     # # #=====================================================================
#     # model.top_layer = None
#
#     # create optimizer
#     optimizer = torch.optim.SGD(
#         filter(lambda x: x.requires_grad, model.parameters()),  # 产生迭代器，x.requires_grad为真带入parameters()
#         lr=0.005,
#     )
#     # define loss function
#     criterion = nn.CrossEntropyLoss()
#
#     # 保存每个epoch的kmeans 和 cnn 的 acc绘制最后的折线图
#     cnn_acc_list = []
#     for epoch in range(EPOCH):
#         end = time.time()
#         # remove head for cnn features
#         # model.top_layer = None
#         # 返回的train_dataset 数据按照真实标签从小到大一组 每组样本下标从小到大排列
#         # train_dataset = clustering.cluster_assign(I, img, 9000)
#
#         # uniformely sample per target   均匀采样(打乱数据去训练cnn）
#         # sampler = util.UnifLabelSampler(len(train_dataset),deepcluster.images_lists)
#         # make train_dataloader -> images_data & pseudolabels
#         # train_dataloader = torch.utils.data.DataLoader(
#         #     trainSet,
#         #     batch_size=BATCH_SIZE,
#         #     # sampler=sampler,
#         #     pin_memory=True,
#         # )
#         model.to(device)
#         # train network with clusters as pseudo-labels
#         print_fn('train cnn models...')
#         train(trainloader, model, criterion, optimizer, epoch)
#
#         print_fn('Epoch:{}  Time: {}s'.format(epoch, time.time()-end))
#
#
#         c_acc = cnn_acc(model, testloader)
#         cnn_acc_list.append(c_acc)
#         print_fn('Epoch:{}    cnn accuray:{}'.format(epoch, c_acc))
#
#     # 展示准确度变化曲线
#     show_acc_line(cnn_acc_list)
#     # 保存最终的模型
#     torch.save(model.state_dict(), finalmodel)
#     print_fn('cnn model is saved to {}'.format(finalmodel))


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
                # print_fn('[%d, %d] loss: %.03f'
                #       % (epoch + 1, i + 1, sum_loss / 10))
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
            print_fn('第%d个epoch，train上识别准确率为：%.3f%%, test上识别准确率为：%.3f%%'
                  % (epoch + 1,acc_train, acc))

            if acc > bast_acc: # 保存泛化能力最好的模型
                bast_acc = acc
                bast_model = net.state_dict()

    # 保存模型参数
    torch.save(bast_model, model_save_path)
    print_fn('model is saved to {}'.format(model_save_path))

def main():
    # 超参数设置
    EPOCH = 200  # 遍历数据集次数
    BATCH_SIZE = 1024  # 批处理尺寸(batch_size)
    LR = 0.0001  # 学习率
    num_clus = 3  # CNN分类器分类数量

    model_save_path = save_path + 'parameter64.pth'

    # 加载频谱数据
    trainloader, testloader = load_data(BATCH_SIZE)
    print_fn('premodel data size:{}'.format(trainloader.dataset.x_data.shape))

    # 训练模型
    net = models.alexnet(out=num_clus).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, net.parameters()),  # 产生迭代器，x.requires_grad为真带入parameters()
        lr=LR,
    )

    # print_fn(net)
    train(net, optimizer, criterion, EPOCH, trainloader, testloader, model_save_path)


# 测试cnn 在dataloader上的预测准确度
def cnn_acc(model, dataloader):
    model.eval()
    pred_labels = []
    true_labels = []
    for i, data in enumerate(dataloader):
        input_tensor, target = data
        input_var = torch.autograd.Variable(input_tensor).cuda()
        true_labels += list(target)
        with torch.no_grad():
            output = model(input_var)
        pred = torch.max(output, 1)[1]
        pred_labels += list(pred.cpu().numpy())

    acc = accuracy_score(true_labels, pred_labels)
    return acc


# 绘制cnn kmeans 准确度变化曲线
def show_acc_line(c_acc, k_acc):
    acc=[c_acc,k_acc]
    np.save(save_path + 'cnn_kmean_acc_wopretrain.npy', acc)
    y1 = [data*100 for data in c_acc]
    y2 = [data*100 for data in k_acc]
    x = range(0, len(c_acc))
    plt.plot(x, y1, color="r", label='cnn')
    plt.plot(x, y2, color="b", label='kmeans')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # 显示表格
    plt.grid(alpha=0.4, linestyle=':')
    plt.legend()
    plt.savefig(save_path + "cnn_kmeans.png",bbox_inches='tight', pad_inches = -0.1)
    # plt.show()


if __name__ == '__main__':
    main()
