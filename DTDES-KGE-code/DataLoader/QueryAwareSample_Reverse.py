#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import torch
import time
import random
import os

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import itertools

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, args=None):
        self.args = args
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_triples = self.get_true_head_and_tail(self.triples)
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        if 'FB15k-237' in self.args.data_path:
            self.query_aware_dict1, self.query_aware_dict2 = self.read_qad(self.args.data_path + '/RotatE_LorentzKG_qt_dict.txt')
        if 'wn18rr' in self.args.data_path:
            self.query_aware_dict1, self.query_aware_dict2 = self.read_qad(self.args.data_path + '/HAKE_LorentzKG_qt_dict.txt')
        self.relation_aware_dict = self.read_rad(self.args.data_path + '/rt_dict.txt')
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        new_seed = int(time.time() * 1000) % 10000 
        np.random.seed(new_seed) # 重置随机种子
        
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample
        
        # 出现次数少权重就大，出现次数多权重就少。
        subsampling_weight = self.count[(head, relation)]
        
        if relation >= self.nrelation/2: # 说明这个三元组是reverse之后的
            subsampling_weight += self.count[(tail, relation-int(self.nrelation/2))]
        else: # 说明这个三元组是reverse之前的
            subsampling_weight += self.count[(tail, relation+int(self.nrelation/2))]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        if self.args.pre_sample_size == 0:
            negative_sample_list = []
            negative_sample_size = 0
        else:
            # pre_sample_list = self.pre_sampling1(head, relation, tail, pre_sample_num=self.args.pre_sample_size) # 从第一个教师模型采样
            pre_sample_list = self.pre_sampling2(head, relation, tail, pre_sample_num=self.args.pre_sample_size) # 从第二个教师模型采样
            # pre_sample_list = self.pre_sampling3(head, relation, tail, pre_sample_num=self.args.pre_sample_size) # 从两个教师模型采样
            negative_sample_list = [pre_sample_list]
            negative_sample_size = len(pre_sample_list)
        
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.args.pre_sample_size != 0:
                exclude_set = np.intersect1d(
                    self.query_aware_dict2[(head, relation)], 
                    self.true_triples[(head, relation)], 
                    assume_unique=False
                )
            else:
                exclude_set = self.true_triples[(head, relation)]

            mask = np.in1d(
                negative_sample, 
                exclude_set,
                assume_unique=False, 
                invert=True 
            )
            # 去除掉对应位置为False的样本
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        # 去掉多余的
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        np.random.shuffle(negative_sample)
        

        if head not in self.true_triples[(head, relation)]:
            negative_sample[random.randint(0, self.negative_sample_size - 1)] = head

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        
        return positive_sample, negative_sample, subsampling_weight, 'QueryAwareSample'
    
    def read_rad(self, path):
        result = {}
        try:
            with open(path, 'r') as file:
                for line in file:
                    parts = line.strip().split()  # Split the line into parts by whitespace
                    if parts:  # Ensure the line is not empty
                        key = int(parts[0])  # The first number as the key
                        values = list(map(int, parts[1:]))  # The rest as a list of integers
                        result[key] = np.array(np.unique(values))
        except FileNotFoundError:
            print(f"警告：文件 {path} 不存在，返回空字典")
            return {}
        
        return result
    
    def read_qad(self, path):
        PT1_result = {}
        PT2_result = {}
        
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
                for i in range(0, len(lines), 3):
                    head_relation = lines[i].strip().split("\t")
                    head, relation = int(head_relation[0]), int(head_relation[1])

                    pt1_ids = list(map(int, lines[i + 1].strip().split("\t")))
                    pt2_ids = list(map(int, lines[i + 2].strip().split("\t")))

                    PT1_result[(head, relation)] = np.array(np.unique(pt1_ids))
                    PT2_result[(head, relation)] = np.array(np.unique(pt2_ids))
        except FileNotFoundError:
            print(f"警告：文件 {path} 不存在，返回空字典")
            return {}, {}
        
        return PT1_result, PT2_result
    
    def relation_pre_sampling(self, head, relation, tail, RAS=30, invalid_ids=None):
        relation_aware_tail = self.relation_aware_dict[relation]
        mask = np.in1d(
            relation_aware_tail, 
            np.intersect1d(invalid_ids, self.true_triples[(head, relation)], assume_unique=True),
            assume_unique=True, 
            invert=True #返回一个和negative_sampleU一样大的数组True表示negative_sample中不在self.true_head[(relation, tail)]中的元素
        )
        relation_aware_tail = relation_aware_tail[mask]
        np.random.shuffle(relation_aware_tail)
        
        selected_tail = relation_aware_tail[:RAS]
        return selected_tail
    
    def pre_sampling1(self, head, relation, tail, pre_sample_num=50):
        query_aware_tail = self.query_aware_dict1[(head, relation)]
        mask = np.in1d(
            query_aware_tail, 
            np.append(self.true_triples[(head, relation)], head),
            assume_unique=True, 
            invert=True #返回一个和negative_sampleU一样大的数组True表示negative_sample中不在self.true_head[(relation, tail)]中的元素
        )
        query_aware_tail = query_aware_tail[mask]
        np.random.shuffle(query_aware_tail)
        
        selected_tail = query_aware_tail[:pre_sample_num]
        return selected_tail
    
    
    def pre_sampling2(self, head, relation, tail, pre_sample_num=50):
        query_aware_tail = self.query_aware_dict2[(head, relation)]
        mask = np.in1d(
            query_aware_tail, 
            np.append(self.true_triples[(head, relation)], head),
            assume_unique=True, 
            invert=True #返回一个和negative_sampleU一样大的数组True表示negative_sample中不在self.true_head[(relation, tail)]中的元素
        )
        query_aware_tail = query_aware_tail[mask]
        np.random.shuffle(query_aware_tail)
        
        selected_tail = query_aware_tail[:pre_sample_num]
        return selected_tail
    
    def pre_sampling3(self, head, relation, tail, pre_sample_num=50):
        query_aware_tail1 = self.query_aware_dict1[(head, relation)]
        query_aware_tail2 = self.query_aware_dict2[(head, relation)]
        
        query_aware_tail = np.unique(np.concatenate([query_aware_tail1, query_aware_tail2]))
        mask = np.in1d(
            query_aware_tail, 
            np.append(self.true_triples[(head, relation)], head),
            assume_unique=True, 
            invert=True #返回一个和negative_sampleU一样大的数组True表示negative_sample中不在self.true_head[(relation, tail)]中的元素
        )
        query_aware_tail = query_aware_tail[mask]
        np.random.shuffle(query_aware_tail)
        
        selected_tail = query_aware_tail[:pre_sample_num]
        return selected_tail
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_triples = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_triples:
                true_triples[(head, relation)] = []
            true_triples[(head, relation)].append(tail)
        # 去重后返回
        for head, relation in true_triples:
            true_triples[(head, relation)] = np.array(list(set(true_triples[(head, relation)])))             

        return true_triples
    

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]


        tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
        tmp[tail] = (0, tail)

        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        
        return positive_sample, negative_sample, filter_bias




class BidirectionalOneShotIterator(object):
    def __init__(self, *train_dataloaders):
        if len(train_dataloaders) == 1:
            self.train_dataloader = iter(train_dataloaders[0])  # 将DataLoader转换为迭代器
            self.iterating_from_second = False  # 只有一个数据集时，不需要切换
        elif len(train_dataloaders) == 2:
            self.dataloader1 = iter(train_dataloaders[0])  # 将DataLoader转换为迭代器
            self.dataloader2 = iter(train_dataloaders[1])  # 将DataLoader转换为迭代器
            self.iterating_from_second = False  # 初始时从第一个数据集开始
        else:
            raise ValueError("Only 1 or 2 dataloaders are supported.")
        
        # 保存初始化的参数，用于后续重新创建DataLoader
        self.datasets = [dataloader.dataset for dataloader in train_dataloaders]
        self.args = [dataloader.batch_size for dataloader in train_dataloaders]  # batch_size参数
        self.num_workers = [dataloader.num_workers for dataloader in train_dataloaders]  # num_workers
        self.collate_fn = [dataloader.collate_fn for dataloader in train_dataloaders]  # collate_fn
        
        self.dataset1_iternum = 1
        self.dataset2_iternum = 1
        self.counter1 = 0
        self.counter2 = 0
        
        # 计算总长度
        if len(train_dataloaders) == 1:
            self.total_length = len(train_dataloaders[0])
        elif len(train_dataloaders) == 2:
            self.total_length = sum([len(train_dataloaders[0])*self.dataset1_iternum, len(train_dataloaders[1])*self.dataset2_iternum])
        
        

    def __next__(self):
        if hasattr(self, 'dataloader1') and hasattr(self, 'dataloader2'):
            # 检查当前采样次数是否已经达到设定的比例
            if self.counter1 < self.dataset1_iternum:
                # 从第一个数据集采样
                try:
                    data = next(self.dataloader1)
                    self.counter1 += 1
                except StopIteration:
                    # 第一个数据集迭代完，重新创建并shuffle
                    self.dataloader1 = self.create_dataloader(self.datasets[0], 0)
                    self.dataloader1 = iter(self.dataloader1)  # 再次转换为迭代器
                    self.counter1 = 0
                    data = next(self.dataloader1)
                    self.counter1 += 1
            elif self.counter2 < self.dataset2_iternum:
                # 从第二个数据集采样
                try:
                    data = next(self.dataloader2)
                    self.counter2 += 1
                except StopIteration:
                    # 第二个数据集迭代完，重新创建并shuffle
                    self.dataloader2 = self.create_dataloader(self.datasets[1], 1)
                    self.dataloader2 = iter(self.dataloader2)  # 再次转换为迭代器
                    self.counter2 = 0
                    data = next(self.dataloader2)
                    self.counter2 += 1
            else:
                # 两个数据集都完成了当前比例的迭代，重置计数器
                self.counter1 = 0
                self.counter2 = 0
                # 重新开始采样，第一个数据集优先
                data = next(self)
        else:
            # 如果只有一个数据集，沿用当前的做法
            try:
                data = next(self.train_dataloader)
            except StopIteration:
                # 迭代完后重新创建并shuffle
                self.train_dataloader = self.create_dataloader(self.datasets[0], 0)
                self.train_dataloader = iter(self.train_dataloader)  # 转换为迭代器
                data = next(self.train_dataloader)
        
        return data
    
    def create_dataloader(self, dataset, index):
        """
        创建一个新的DataLoader并对数据集进行shuffle。
        """
        return DataLoader(
            dataset,
            batch_size=self.args[index],  # batch_size从初始化时记录
            shuffle=True,              # 保证shuffle
            num_workers=self.num_workers[index],  # num_workers从初始化时记录
            collate_fn=self.collate_fn[index]     # collate_fn从初始化时记录
        )

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data



"""
这个迭代器允许有两个风格不一致的训练数据集，并且交替的取出其中的数据
"""
class BidirectionalOneShotIterator2(object):
    def __init__(self, train_dataloader1, train_dataloader2):
        self.train_dataloader1 = self.one_shot_iterator(train_dataloader1)
        self.train_dataloader2 = self.one_shot_iterator(train_dataloader2)
        self.step = 0
        self.len = len(train_dataloader1) + len(train_dataloader2)
        
    def __next__(self):
        # 轮流替换头和尾
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.train_dataloader1)
        else:
            data = next(self.train_dataloader2)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data



"""
简单的 单一数据集的迭代器
"""
class SimpleBidirectionalOneShotIterator(object):
    def __init__(self, train_dataloader):
        self.train_dataloader = self.one_shot_iterator(train_dataloader)
        self.total_length = len(train_dataloader)
        
    def __next__(self):
        # 轮流替换头和尾
        data = next(self.train_dataloader)

        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data