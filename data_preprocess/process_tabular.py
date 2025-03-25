import torch
import numpy as np
import scipy.io
import os
import sys
import csv

def train_test_split(inliers, outliers):
    inliers = inliers[np.random.permutation(inliers.shape[0])]
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    test_data = np.concatenate([inliers[num_split:], outliers], 0)

    test_label = np.zeros(test_data.shape[0])
    test_label[inliers[num_split:].shape[0]:] = 1
    return train_data, test_data, test_label

class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains

    def get_dataset(self, dataset_name):
        script_dir = os.path.dirname(__file__)
        rel_path = os.path.join("data/", dataset_name)
        abs_file_path = os.path.join(script_dir, rel_path)
        mat_files = ['arrhythmia', 'breastw', 'cardio', 'glass', 'ionosphere', 'mammography', 'pendigits', 'pima', 'satellite', 'satimage', 'shuttle', 'thyroid', 'wbc', 'wine', 'optdigits']
        if dataset_name in mat_files:
            print('generic mat file')
            return self.build_train_test_generic_matfile(abs_file_path)

        elif dataset_name in ['census', 'campaign', 'cardiotocography', 'fraud']:
            return self.build_npz_dataset(abs_file_path + '.npz')

        elif dataset_name in ['nslkdd']:
            return self.build_csv_dataset(abs_file_path + '.csv')

        sys.exit('No such dataset!')

    def build_csv_dataset(self, name_of_file):
        x = []
        labels = []
        with open(name_of_file, 'r') as data_from:
            csv_reader = csv.reader(data_from)
            for i in csv_reader:
                x.append(i[0:122])
                labels.append(i[122])

        for i in range(len(x)):
            for j in range(122):
                x[i][j] = float(x[i][j])
        for i in range(len(labels)):
            labels[i] = float(labels[i])

        data = np.array(x)
        target = np.array(labels)
        inlier_indices = np.where(target == 0)[0]
        outlier_inices = np.where(target == 1)[0]
        train_data, test_data, test_label = train_test_split(data[inlier_indices], data[outlier_inices])

        train_data = torch.tensor(train_data)
        test_data = torch.tensor(test_data)
        test_label = torch.tensor(test_label)

        return (train_data, test_data, test_label)


    def build_npz_dataset(self, name_of_file):
        data = np.load(name_of_file)
        samples = data['X']
        labels = ((data['y']).astype(np.int64)).reshape(-1)
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, test_data, test_label = train_test_split(inliers, outliers)
        train_data = torch.tensor(train_data)
        test_data = torch.tensor(test_data)
        test_label = torch.tensor(test_label)

        return (train_data, test_data, test_label)


    def build_train_test_generic_matfile(self, name_of_file):
        dataset = scipy.io.loadmat(name_of_file)
        X = dataset['X']
        classes = dataset['y']
        jointXY = torch.cat((torch.tensor(X, dtype=torch.double), torch.tensor(classes, dtype=torch.double)), dim=1)
        normals = jointXY[jointXY[:, -1] == 0]
        anomalies = jointXY[jointXY[:, -1] == 1]
        normals = normals[torch.randperm(normals.shape[0])]
        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0:train.shape[1] - 1]
        test = test[:, 0:test.shape[1] - 1]

        return (train, test, test_classes)
