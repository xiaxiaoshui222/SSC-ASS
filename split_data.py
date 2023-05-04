import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
import fnmatch
import numpy as np
from scipy.io import loadmat

def load_data(flag,db, directory, unlabel_size=0.2):

    directorys = []
    if flag :
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if fnmatch.fnmatch(filename, db +".npy"):
                    directorys.append(os.path.join(root, filename))

                    print(os.path.join(root, filename))

        for i, dirs in enumerate(directorys):
            matdata = np.load(dirs)
            if i == 0:
                res = matdata
            else:
                res = np.concatenate((res, matdata), axis=0)
    else:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if fnmatch.fnmatch(filename, "data.npy"):
                    if db == os.path.basename(os.path.dirname(os.path.join(root, filename))):
                        directorys.append(os.path.join(root, filename))

                        print(os.path.join(root, filename))

        for i, dirs in enumerate(directorys):
            matdata = np.load(dirs)
            if i == 0:
                res = matdata
            else:
                res = np.concatenate((res, matdata), axis=0)

    before_len = 28
    after_len = 28
    # print(res.shape)
    # res = np.pad(res, ((0, 0), (before_len, after_len), (0, 0)), 'constant')
    print(res.shape)
    # np.save('./data/hecheng0-data.npy', res.reshape(36000, 1, 64, 64))

    print('init train_data and test_data')
    dataset = res
    # dataset = np.load('./data/-5-data.npy')

    print("dataset.size", dataset.shape)

    label = []
    for i in range(3):
        l = [i] * 2000
        label += l
    label = np.array(label)

    state = np.random.get_state()
    np.random.shuffle(dataset)
    np.random.set_state(state)
    np.random.shuffle(label)
    train_data, test_data, train_label, test_label = train_test_split(dataset, label, test_size=unlabel_size)
    save_path = os.path.join('train_sample', directory, db)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + '/train_data.npy', train_data)
    np.save(save_path + '/test_data.npy', test_data)
    np.save(save_path + '/train_label.npy', train_label)
    np.save(save_path + '/test_label.npy', test_label)

# arg1 = sys.argv[1]
# print(arg1)
arg1 = '0'
load_data(False, arg1, 'zhenshi')