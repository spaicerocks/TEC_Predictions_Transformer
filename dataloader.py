from random import shuffle
import torch
import csv
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset, SequentialSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
from joblib import dump, load 
import math
import os
from sklearn.preprocessing import MinMaxScaler
from GIM_TXT_to_csv import read_file, natural_keys, read_omni

def combine_value(tec_map, omni):
    tec_map, dst_map = tec_map[0], [[omni[0][0] for x in range(72)] for y in range(71)]
    f_map, kp_map = [[omni[0][1] for x in range(72)] for y in range(71)], [[omni[0][2] for x in range(72)] for y in range(71)]
    final_map = []
    for a, b, c, d in zip(tec_map, dst_map, f_map, kp_map):
        final_map.append([x for y in zip(a, b, c, d) for x in y])
    return final_map

def get_data(path):
    window_size = 1
    _input, target, date = [], [], []
    full_data, datetimes, omnis = [], [], []
    file_list = os.listdir(path)
    
    for file_name in file_list:
        one_day_data, datetime, omni = read_file(path+file_name)
        full_data += one_day_data
        datetimes += datetime
        omnis += omni
    for idx in range(len(full_data)):
        if idx + window_size >= len(full_data):break
        #final_map = combine_value(full_data[idx:idx+window_size], omnis[idx:idx+window_size])
        _input.append(full_data[idx:idx+window_size])
        target.append(full_data[idx+window_size])
        date.append(datetimes[idx+window_size])
        
    return np.array(_input, dtype=np.float64).squeeze(), np.array(target), np.array(date)

def get_dataloader(train, validation, test, batch_size):
    train[0], train[1], train[2] = torch.tensor(train[0], dtype=torch.float), torch.tensor(train[1], dtype=torch.float), torch.tensor(train[2])
    validation[0], validation[1], validation[2] = torch.tensor(validation[0], dtype=torch.float), torch.tensor(validation[1], dtype=torch.float), torch.tensor(validation[2])
    test[0], test[1], test[2] = torch.tensor(test[0], dtype=torch.float), torch.tensor(test[1], dtype=torch.float), torch.tensor(test[2])

    # Create the DataLoader for our training set
    train_data = TensorDataset(train[0], train[1], train[2])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(validation[0], validation[1], validation[2])
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    test_data = TensorDataset(test[0], test[1], test[2])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

    return train_dataloader, val_dataloader, test_dataloader

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class TecDataset(Dataset):
    def __init__(self, path, data_type='dir', mode='train', window_size=4, to_sequence=False):
        self.mon_day = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}  
        self.path = path
        self.mode = mode
        self.data_type = data_type
        self.to_sequence = to_sequence
        '''
        self.transform = MinMaxScaler()
        '''
        self.nor_transform = transforms.Compose([            
            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            #transforms.Normalize([0.5,], [0.5,])
            #transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=1),
            ]
        )
        self.unorm = UnNormalize(mean=[0.5,], std=[0.5,])
        self.window_size = window_size
        datetimes = []
        if self.data_type == 'dir':
            full_data = []
            file_list = os.listdir(path)
            file_list.sort(key = natural_keys)
            for file_name in file_list:
                one_day_data, datetime, _ = read_file(path+file_name)
                full_data += one_day_data
                datetimes += datetime
                #if mode == 'train':full_data += self.normalize_meanstd(one_day_data)
                #else:full_data += one_day_data
            self.datetimes = datetimes
            self.tec_data = full_data
        else: 
            self.tec_data, datetime = read_file(path)  
            self.datetimes = datetime     
        
        self._input, self.target, self.tar_date = self.get_data()
        
        #self._input = self.transform.fit_transform(self._input)
       
        '''
        self.shapes = self._input.squeeze().shape
        nsamples, nx, ny = tuple(self.shapes)
        d2_input = self._input.squeeze().reshape((nsamples,nx*ny))
        self.transform.fit(d2_input)
        self._input = self.transform.transform(self._input.squeeze().reshape((nsamples,nx*ny))).reshape((nsamples,nx,ny)) #將GIM MAP concate在一起
        dump(self.transform, 'scalar_item.joblib')
        #self.target = self.transform.transform(self.target.squeeze().reshape((nsamples,nx*ny))).reshape((nsamples,nx,ny))
        '''

    def normalize_meanstd(self, a, axis=None): 
        # axis param denotes axes along which mean & std reductions are to be performed
        mean = np.mean(a, axis=axis, keepdims=True)
        std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
        return (a - mean) / std

    def mini_max_norm(self, full_data):
        '''
        MaxTEC = max(max(DataTrain));
        MinTEC = 0;
        DataTrainNormal = (DataTrain- MinTEC) ./ (MaxTEC-MinTEC);
        '''
        pass

    def standardize(self, fulldata):
        mean, std = np.mean(data), np.std(data)
        tmp = (data-mean) / std
        return tmp
    
    def omni_std(self, data):
        mean, std = np.mean(data), np.std(data)
        tmp = (data-mean) / std
        return tmp

    def get_data(self):
        _input, target, date = [], [], []
        
        for idx in range(len(self.tec_data)):
            if idx + self.window_size>= len(self.tec_data):break
            #_input.append(self.normalize_meanstd(self.tec_data[idx:idx+self.window_size]))
            _input.append(self.tec_data[idx])
            target.append(self.tec_data[idx+self.window_size])
            date.append(self.datetimes[idx+self.window_size])
        return np.array(_input).squeeze(), np.array(target), date

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):   
        latitude_range = np.linspace(87.5, -87.5, 71) #將緯度依2.5度切割為71等分
        lat = torch.tensor([[math.sin(i/87.5), math.cos(i/87.5)] for i in latitude_range]) #取sin, cos來表示緯度位置        
        #_input = torch.tensor(self._input.squeeze(), dtype=torch.float).squeeze()
        #target = torch.tensor(self.target[idx], dtype=torch.float).squeeze()        
        _input = torch.tensor(self._input[idx], dtype=torch.float)
        target = torch.tensor(self.target[idx], dtype=torch.float)
        day = sum([self.mon_day[i] for i in range(1,self.tar_date[idx][1])]) + self.tar_date[idx][2] 
        tar_date = torch.tensor([[math.sin((day)/366), math.cos(day/366), math.sin(self.tar_date[idx][3]/24), \
        math.cos(self.tar_date[idx][3]/24)]for i in range(71)], dtype=torch.float)
        information = torch.cat((lat, tar_date), 1)
        #_input = torch.cat((_input, lat, tar_date), 1)
        #_input = torch.tensor(self._input[idx], dtype=torch.float)
        return _input, target, self.tar_date[idx], information

if __name__ == '__main__':
    
    #scaler = load('scalar_item.joblib')
    tmpdata = TecDataset('txt/test/', data_type='dir', window_size=1, to_sequence=True)
    #tmpdataloader = DataLoader(tmpdata, batch_size = 16, shuffle = False)    
    #tmpdata = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test', pred_future=True)
    tmpdataloader = DataLoader(tmpdata, batch_size = 64, shuffle = True) 
    
    
    for inp_map, tar_map, date, information in tmpdataloader:
        print(inp_map.size(), tar_map.size(), information.size())
        print(inp_map, tar_map)
        #print(scaler.inverse_transform(tar_map[0].cpu()))
        #print(scaler.inverse_transform(inp_map[0].cpu()))
        #print(date)
        input()
    