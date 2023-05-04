import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def preprocess_features(data, n_pca = 256):
    data = data.astype('float32')
    # Apply PCA-whitening
    pca = PCA(256, whiten=True)
    pca.fit(data)
    data = pca.transform(data)
    return data

def run_kmeans(x, num_cluster=12, centers=None):
    if centers is None:
        k = KMeans(n_clusters=num_cluster, max_iter=100,random_state=123,)
    else:
        centers = np.array(centers)
        k = KMeans(n_clusters=num_cluster, max_iter=100,  random_state=123, n_init=1, init=centers)
    k.fit(x)
    centers_ = k.cluster_centers_
    pseudolabels = k.predict(x)
    return pseudolabels, centers_

class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes      样本下标
        pseudolabels (list): list of labels for each    聚类结果
        dataset (list): list of tuples                  (data,label)顺序是初始数据输入顺序
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, true_size, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset, true_size)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset, true_size):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}  # len(labels_to_index)=12
        images = []
        for j, idx in enumerate(image_indexes):  # len(image_indexes) = N
            imgdata = dataset[idx][0]
            if idx >= true_size:
                pseudolabel = label_to_idx[pseudolabels[j]]
            else:
                pseudolabel = dataset[idx][1].item()
            images.append((imgdata, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        # img = pil_loader(path)
        img = path
        if self.transform is not None:
            img = self.transform(path)
        img = np.expand_dims(img, axis=0)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

def cluster_assign(images_lists, dataset, true_size):
    '''
    images_lists:每个cluster中的样本在dataset中的下表  len = 12
    dataset: 所有的输入样本  (data,label) 包含有标签和无标签数据
    true_size: 数据中含有真实标签的样本数量
    '''
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):  # len(images_lists) == 12
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    # t = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize])
    t = None

    # image_indexes --> (下标，标签)  某个样本在dataset中的位置（下标）
    return ReassignedDataset(image_indexes, pseudolabels, dataset, true_size, t)


class Kmeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, data, centers=None):
        # PCA-reducing, whitening
        # data = preprocess_features(data)
        I, centers_ = run_kmeans(data, self.k, centers)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(I)):
            self.images_lists[I[i]].append(i)

        return self.images_lists, centers_        #images_lists -> [[2, 3, 4], [5, 6], [1], [0]]

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    # print("image_indexes", image_indexes)
    # print("pseudolabels", pseudolabels)
    # print("indexes", indexes)
    print(len(np.asarray(pseudolabels)[indexes]))
    return np.asarray(pseudolabels)[indexes]
    # pseudolabels  -> [0, 0, 0, 1, 1, 2, 3]
    # image_indexes -> [2, 3, 4, 5, 6, 1, 0]
