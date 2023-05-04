import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

from semilearn.core.utils import get_logger
import util
import models
import clustering
import sys
arg1 = sys.argv[1]
db = arg1
data_type = 'chonggou'
save_path = 'savemodels/'+ data_type +'/' + db

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
    train_data = np.load('train_sample/'+ data_type +'/'+ db +'/train_data.npy').reshape(-1, 1 ,64 ,64)
    test_data = np.load('train_sample/'+ data_type +'/'+ db +'/test_data.npy').reshape(-1, 1 ,64 ,64)
    train_label = np.load('train_sample/'+ data_type +'/'+ db +'/train_label.npy')
    test_label = np.load('train_sample/'+ data_type + '/'+ db +'/test_label.npy')

    # 定义频谱数据  testSet作为无标签数据
    testSet = MyDataSet(test_data, test_label)
    testloader = Data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)
    trainSet = MyDataSet(train_data, train_label)
    trainloader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    # 加载全部数据  将train当做有标签  test当做无标签
    dataset = np.concatenate((train_data, test_data), axis=0)
    label = np.concatenate((train_label, test_label), axis=0)
    alldataset = MyDataSet(dataset, label)
    alldataloader = Data.DataLoader(alldataset, batch_size=BATCH_SIZE, shuffle=False)

    print_fn('labeld:{}'.format(train_data.shape[0]))
    print_fn('unlabeled:{}'.format(testSet.len))
    print_fn('alldataset:{}'.format(alldataset.len))

    return alldataset, alldataloader, trainSet, trainloader, testSet, testloader, 'parameter64.pth', 'parameter64final.pth'


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


def main():
    nums_clu = 3
    BATCH_SIZE = 256
    EPOCH = 200

    # load data
    alldataset, alldataloader, trainSet, trainloader, testset, testloader, premodel, finalmodel = load_data()
    finalmodel = save_path + finalmodel
    # CNN
    model = models.alexnet(out=nums_clu)
    # print_fn(model)
    model.to(device)
    # # 加载预训练模型=======================================================
    # print_fn('load pre-trained model form:', premodel)
    para = torch.load(premodel)
    model.load_state_dict(para)
    acc = cnn_acc(model, testloader)
    print_fn('pre-trained model on unlabeled acc:{}'.format(acc))
    # # #=====================================================================

    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),  # 产生迭代器，x.requires_grad为真带入parameters()
        lr=0.005,
    )
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # clustering algorithm to use
    deepcluster = clustering.Kmeans(nums_clu)
    deepcluster_test = clustering.Kmeans(nums_clu)
    # creating cluster assignments log
    cluster_log = util.Logger(os.path.join('clusters'))  # 保存每个epoch的聚类结果  为了计算MNI
    center_log = util.Logger(os.path.join('centers'))  # 保存每个epoch的KMeans质心  为了稳定KMeans聚类效果

    # training conv-net   img--> [(data,lable) x N]  img样本顺序和alldataset样本顺序一致
    img = util.to_PIL(alldataset)
    # 保存每个epoch的kmeans 和 cnn 的 acc绘制最后的折线图
    kmeans_acc_list = []
    cnn_acc_list = []
    ari_list = []
    ari = 0
    for epoch in range(EPOCH):
        end = time.time()
        # remove head for cnn features
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # 定义那个epoch需要绘制聚类效果
        # if epoch in (0, 1, 99, EPOCH - 1):
        #     show_clu = True
        # else:
        #     show_clu = False
        # 给kmeans初始质心
        if epoch > 0:
            centers = center_log.data[-1]
        else:
            centers = None

        features, printlabels = compute_features(model, alldataloader,epoch, centers, show_clu=False)
        # features_test, printlabels_test = compute_features_test(model, testloader, epoch, centers, show_clu=show_clu)
        # break
        # epoch=0 抽样训练cnn数据的前1000个样本的features，计算每类的均值作为质心
        if centers is None:
            centers = []
            for i in range(nums_clu):
                # 取出真实标签为i的所有特征
                features_for_true = [data for index, data in enumerate(features[:1000]) if printlabels[index] == i]
                # 按列求均值作为质心
                centers.append(np.mean(features_for_true, axis=0))
        # print_fn("centets:", centers)
        print_fn('Run KMeans on all_data features...')
        I, centers_ = deepcluster.cluster(features, centers)
        # I_test, _ = deepcluster_test.cluster(features_test, centers_)
        # # 将kmeans的簇类和相应质心 进行调整

        resetLabel = []  # resetLabel存储第i个聚类类别对应的真实标签
        # 计算每个含有标签的真实类别的 memory 方便 KMeans 去match
        # 更新memory 因为cnn模型在逐渐变化，所以计算出来的特征也在逐渐变化，
        # 每类的真正质心也在逐渐变化，仅使用初始memory并不适用改变了的cnn提取出来的特征
        # 是使用上一次的质心还是在所有类别中重新提取memory
        # todo 此处的alldataloader 是不是应该换成仅包含训练集上提取memorys
        memorys = compute_memorys(model, alldataloader, nums_clu, BATCH_SIZE)


        dis = euclidean_distances(centers_, memorys)
        print_fn(dis)
        resort = np.sort(np.array(dis).flatten())
        resetLabel = [-1] * len(centers)
        for i_min in resort:
            cen_c, mem_c = np.where(dis == i_min)
            for c, m in zip(cen_c, mem_c):
                if resetLabel[c] == -1 and m not in resetLabel:
                    resetLabel[c] = m
            if -1 not in resetLabel:
                break;

        # 交换I中元素的位置，实现重新标号的效果  以及相应的质心   I --> [[inx,idx,][]...]  第i个cluster中的数据点在img中的下标
        tempI = [[] for i in range(nums_clu)]
        tempCenters = [[] for i in range(nums_clu)]
        print_fn('resetLabel:{}'.format(resetLabel))
        for i in range(nums_clu):
            tempI[i] = I[resetLabel.index(i)]
            tempCenters[i] = centers_[resetLabel.index(i)]
        I = tempI
        centers_ = tempCenters

        # 保存当前聚类中心，固定每次kmeans簇
        center_log.log(centers_)

        # assign the pseudolaels   img中的数据顺序和导入的npy数据顺序一致
        # 返回的train_dataset 数据按照真实标签从小到大一组 每组样本下标从小到大排列
        train_dataset = clustering.cluster_assign(I, img, 9000)

        # uniformely sample per target   均匀采样(打乱数据去训练cnn）
        sampler = util.UnifLabelSampler(len(train_dataset),deepcluster.images_lists)
        # make train_dataloader -> images_data & pseudolabels
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            pin_memory=True,
        )

        # 添加最后一层全连接层 完成cnn的训练
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True))
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.to(device)
        # train network with clusters as pseudo-labels
        print_fn('train cnn models...')
        train(train_dataloader, model, criterion, optimizer, epoch)
        if epoch > 0:
            # 比较的是所有数据的NMI有标签和无标签的NMI  无标签数据较少，所以变化小
            nmi = normalized_mutual_info_score(
                clustering.arrange_clustering(deepcluster.images_lists),
                clustering.arrange_clustering(cluster_log.data[-1])
            )
            # ARI = adjusted_rand_score(y_true,y_pred)
            ari = adjusted_rand_score(
                clustering.arrange_clustering(deepcluster.images_lists),
                clustering.arrange_clustering(cluster_log.data[-1])
            )
            print_fn('NMI against previous assignment: {0:.3f}'.format(nmi))
            print_fn('ARI against previous assignment: {0:.3f}'.format(ari))
        # log the cluster assignment
        cluster_log.log(deepcluster.images_lists)

        print_fn('Epoch:{}  Time: {}s'.format(epoch, time.time()-end))
        # features_test, printlabels_test = compute_features_test(model, testloader, epoch, centers_, show_clu=show_clu)
        # I_test, _ = deepcluster_test.cluster(features_test, centers_)

        last_cluster = clustering.arrange_clustering(deepcluster.images_lists)
        unlabeled_data = testset.y_data
        k_acc = accuracy_score(unlabeled_data, last_cluster[28800:])
        c_acc = cnn_acc(model, testloader)
        kmeans_acc_list.append(k_acc)
        cnn_acc_list.append(c_acc)
        ari_list.append(ari)
        print_fn('Epoch:{} KMeans accuray:{}    cnn accuray:{}'.format(epoch, k_acc, c_acc))

    # 展示准确度变化曲线
    show_acc_line(cnn_acc_list,kmeans_acc_list)
    np.save(save_path + 'ari_lsit-nopre.npy', ari_list)
    show_arr(ari_list)
    # 保存最终的模型
    torch.save(model.state_dict(), finalmodel)
    print_fn('cnn model is saved to {}'.format(finalmodel))


# 对打完伪标签的数据进行cnn训练
def train(loader, model, crit, opt, epoch):
    # record & update losses
    losses = util.AverageMeter()
    # switch to train mode
    model.train()
    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.001,
    )

    for i, (input_tensor, target) in enumerate(loader):         # loader->train_loader, target为fake labels
        input_var = torch.autograd.Variable(input_tensor).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        output = model(input_var)
        loss = crit(output, target_var)
        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))
        # compute gradient and do SGD step
        opt.zero_grad()  # opt->参数优化器
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()
        if i % 100 == 99:
            print_fn('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i+1, len(loader),loss=losses))
    # return losses.avg

def show_arr(ari):
    size = 10.5
    x = range(0, len(ari))
    plt.ylim([0.8, 1.0])
    fig = plt.figure(figsize=(3, 2.5), dpi=300)
    plt.plot(x, ari, color="r")
    plt.tick_params(labelsize=size)
    plt.xlabel('训练轮次', fontsize=size)
    plt.ylabel('ARI', fontsize=size)
    # plt.legend(loc='lower right', fontsize=size)
    plt.tight_layout()
    plt.savefig(save_path + "ari_cnn.png",bbox_inches='tight', pad_inches = -0.1)
    # plt.show()

# 提取整个数据集的feature
def compute_features(model, trainloader, epoch, centers=None, show_clu=False):
    nums_clu=3
    BATCH_SIZE = 256
    model.eval()
    # 抽取全部训练集特征  忽略标签   features最后保存的是训练集所有的特征  features.shape   (N,512)
    printlabels=[]
    N = len(trainloader.sampler)
    for i, (input_tensor, _) in enumerate(trainloader):
        printlabels += list(_.numpy())
        input_var = torch.autograd.Variable(input_tensor.cuda())
        with torch.no_grad():
            aux = model(input_var).data.cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(trainloader) - 1:
            features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = aux
        else:
            features[i * BATCH_SIZE:] = aux
    print_fn('features.shape:{}'.format(features.shape))

    ############################################################################################################
    if show_clu:  # 对提取到的特征进行聚类查看效果
        show_kmeans_reslut(nums_clu,centers,features,printlabels, epoch)

    return features,printlabels
    #################################################################################################################

def compute_features_test(model, trainloader, epoch, centers=None, show_clu=False):
    nums_clu=3
    BATCH_SIZE = 256
    model.eval()
    # 抽取全部训练集特征  忽略标签   features最后保存的是训练集所有的特征  features.shape   (N,512)
    printlabels=[]
    N = len(trainloader.sampler)
    for i, (input_tensor, _) in enumerate(trainloader):
        printlabels += list(_.numpy())
        input_var = torch.autograd.Variable(input_tensor.cuda())
        with torch.no_grad():
            aux = model(input_var).data.cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(trainloader) - 1:
            features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = aux
        else:
            features[i * BATCH_SIZE:] = aux
    print_fn('features.shape:{}'.format(features.shape))

    ############################################################################################################

    return features,printlabels

# 展示cnn提取到的特征进行kmeans的聚类结果
def show_kmeans_reslut(nums_clu,centers,features,printlabels,it):
    if centers is None:
        centers = []
        # 若当前是epoch0 则选取特征中的每类均值作为初始质心
        # 取前1000个feature特征  计算每个真实标签的均值作为聚类中心
        for i in range(nums_clu):  # 查找第i个特征
            # 取出真实标签为i的所有特征
            features_for_true = [data for index, data in enumerate(features[:1000]) if printlabels[index] == i]
            # 按列求均值作为质心
            centers.append(np.mean(features_for_true, axis=0))

    centers = np.array(centers)
    k = KMeans(n_clusters=nums_clu, max_iter=1000, random_state=1234, init=centers, n_init=1)

    feat = features
    lab = printlabels
    k.fit(feat)
    y_pred = k.predict(feat)

    images_lists = [[] for i in range(nums_clu)]
    for i in range(len(y_pred)):
        images_lists[y_pred[i]].append(i)
    for i in range(nums_clu):
        print_fn(len(images_lists[i]))
    # print_fn()
    color = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: '#FF69B4', 8: '#808080', 9: '#909900',
             10: '#000080', 11: '#8B4513'}
    t = TSNE(random_state=1234)
    point = t.fit_transform(feat)
    count = 0
    plt.figure(figsize=(19.20, 10.80))

    for (reduce_x, reduce_y), j in zip(point, y_pred):
        # plt.plot(reduce_x, reduce_y, color=color[j], marker='o')
        plt.plot(reduce_x, reduce_y, color='w', marker='.')
        # plt.text(reduce_x, reduce_y, str(lab[count]), color=color[lab[count]], fontsize=10)  # 同一颜色为预测出的同一类
        plt.text(reduce_x, reduce_y, str(lab[count]), color=color[j], fontsize=10)  # 同一颜色为预测出的同一类
        count += 1
    #
    plt.xticks([])
    plt.yticks([])
    plt.savefig( save_path + "kmeans-cluster_{}.png".format(it), bbox_inches='tight', pad_inches = -0.1)
    # plt.show()



# 使用cnn去提取loader上所有feature，得到每个真实类别中feature的均值（memory）
def compute_memorys(model, loader, nums_clu, BATCH_SIZE):
    model.eval()
    # 抽取全部训练集特征  忽略标签   features最后保存的是训练集所有的特征  features.shape   (N,512)
    printlabels = []
    N = len(loader.sampler)
    for i, (input_tensor, _) in enumerate(loader):
        printlabels += list(_.numpy())
        input_var = torch.autograd.Variable(input_tensor.cuda())
        with torch.no_grad():
            aux = model(input_var).data.cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(loader) - 1:
            features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = aux
        else:
            features[i * BATCH_SIZE:] = aux
    memorys =[]
    for i in range(nums_clu):
        # 取出真实标签为i的所有特征
        features_idx = [data for index, data in enumerate(features) if printlabels[index] == i]
        # 按列求均值作为质心
        memorys.append(np.mean(features_idx, axis=0))
    return memorys


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
