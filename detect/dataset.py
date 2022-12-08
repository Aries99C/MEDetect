import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MySegLoader(Dataset):
    def __init__(self, win_size, step, mode='train', scaler='standard'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()

        train_data = pd.read_csv('./data/train.csv')
        train_data = train_data.values[:, 1:]
        train_data = np.nan_to_num(train_data)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

        test_data = pd.read_csv('./data/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.valid = self.test

        self.test_labels = pd.read_csv('./data/test_label.csv').values[:, 1:]

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'valid':
            return (self.valid.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'train':
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'valid':
            return np.float32(self.valid[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                   np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(batch_size, win_size=64, step=1, mode='train'):
    data = MySegLoader(win_size=win_size, step=step, mode=mode)
    shuffle = True if mode == 'train' else False
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader
