import argparse
import json
import logging
import os
import random
import copy

import numpy as np
import torch

from torch.utils.data import DataLoader


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Pruning Models',
        usage='Run.py [<args>] [-h | --help]'
    )

    parser.add_argument('-cuda', '--cuda', action='store_true', help='use GPU')
    parser.add_argument('-seed', '--seed', default=42, type=int, help='manual_set_random_seed')
    parser.add_argument('-data_path', '--data_path', type=str, default=None)
    parser.add_argument('-entity_mul', '--entity_mul', type=int, default=1)
    parser.add_argument('-relation_mul', '--relation_mul', type=int, default=1)
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-batch_size', '--batch_size', default=1024, type=int)
    parser.add_argument('-negative_sample_size', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-positive_sample_size', '--positive_sample_size', default=50, type=int)
    parser.add_argument('-pre_sample_size', '--pre_sample_size', default=50, type=int)
    parser.add_argument('-distill_sample_size', '--distill_sample_size', default=50, type=int)
    parser.add_argument('-epoch','--epoch', default=100000, type=int)
    parser.add_argument('-input_dim', '--input_dim', default=512, type=int)
    parser.add_argument('-hidden_dim', '--hidden_dim', default=128, type=int)
    parser.add_argument('-target_dim', '--target_dim', default=None, type=int, help='feature pruning with method')
    parser.add_argument('-pos_gamma', '--pos_gamma', default=12.0, type=float)
    parser.add_argument('-neg_gamma', '--neg_gamma', default=12.0, type=float)
    parser.add_argument('-gammaTrue', '--gammaTrue', default=2.0, type=float)
    parser.add_argument('-stu_rotate_gamma', '--stu_rotate_gamma', default=9.0, type=float)
    parser.add_argument('-reg', '--reg', default=0.0, type=float)
    parser.add_argument('-dropout', '--dropout', default=0.4, type=float)
    parser.add_argument('-negative_adversarial_sampling', '--negative_adversarial_sampling', action='store_true', default=False)
    parser.add_argument('-adversarial_temperature', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-subsampling', '--subsampling', action='store_true', default=False)
    parser.add_argument('-regularization', '--regularization', default=0.0, type=float)
    parser.add_argument('-add_bias', '--add_bias', action='store_true', default=False, help='Add bias (default: True)')
    
    parser.add_argument('-test_batch_size', '--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    
    parser.add_argument('-optimizer', '--optimizer', default='SGD', type=str)
    parser.add_argument('-scheduler', '--scheduler', default='MultiStepLR', type=str)
    parser.add_argument('-momentum', '--momentum', default=0.0, type=float)
    parser.add_argument('-weight_decay', '--weight_decay', default=0.0, type=float)
    parser.add_argument('-nesterov', '--nesterov', action='store_true', default=False) # SGD
    parser.add_argument('-betas', '--betas', default=(0.9, 0.999)) # Adam
    parser.add_argument('-amsgrad', '--amsgrad', action='store_true') # Adam
    parser.add_argument('-patience', '--patience', default=2000, type=int)
    parser.add_argument('-cooldown', '--cooldown', default=2000, type=int)
    parser.add_argument('-warm_up_epochs', '--warm_up_epochs', default=None, type=int)
    parser.add_argument('-decreasing_lr', '--decreasing_lr', default=0.1, type=float)
    
    parser.add_argument('-init_checkpoint', '--init_checkpoint', default='without', type=str)
    parser.add_argument('-save_path', '--save_path', default=None, type=str)
    parser.add_argument('-pretrain_path', '--pretrain_path', type=str, default='without', help='pretrained model path')
    parser.add_argument('-pretrain_path2', '--pretrain_path2', type=str, default='without', help='pretrained model path')
    parser.add_argument('-save_checkpoint_epochs', '--save_checkpoint_epochs', default=10, type=int)
    parser.add_argument('-log_epochs', '--log_epochs', default=1, type=int, help='train log every xx epochs')
    parser.add_argument('-test_per_epochs', '--test_per_epochs', default=10, type=int, help='test every xx epochs')
    parser.add_argument('-test_log_steps', '--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('-nentity', '--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('-nrelation', '--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('-dynamic_weight', '--dynamic_weight', action='store_true', default=False)
    parser.add_argument('-fusion_score_loss_weight', '--fusion_score_loss_weight', type=float, default=1.0)  
    
    # 给SCCF-decoder留的
    parser.add_argument('-temprature', '--temprature', default=0.1, type=float)
    parser.add_argument('-cl_tau', '--cl_tau', default=0.4, type=float)
    parser.add_argument('-sccf_margin', '--sccf_margin', default=6.0, type=float)
    
    # 双曲空间需要得
    parser.add_argument('-init_size', '--init_size', default=0.001, type=float)
    parser.add_argument('-data_type', '--data_type', default='single', type=str)
    
    # 给transformer提取语义特征留的
    parser.add_argument('-heads', '--heads', type=int, default=4, help='Transformer dff dimension')
    
    # 给知识蒸馏损失留的
    parser.add_argument('-stu_pre_process', '--stu_pre_process', default='local_standardize', type=str)
    parser.add_argument('-tea_pre_process', '--tea_pre_process', default='constant', type=str)
    parser.add_argument('-distill_func', '--distill_func', default='KL_divergence', type=str)
    parser.add_argument('-weight_mask', '--weight_mask', default='[0, 1, 0, 1, 0]', type=str)
    parser.add_argument('-temprature_ts', '--temprature_ts', default=1.0, type=float)
    parser.add_argument('-kd_gamma', '--kd_gamma', default=12.0, type=float)
    parser.add_argument('-kdloss_weight', '--kdloss_weight', type=float, default=0.01, help='KD loss weight')
    
    # 给对比学习的
    parser.add_argument('-contrastive_tau', '--contrastive_tau', default=1.0, type=float)
    parser.add_argument('-contrastive_weight', '--contrastive_weight', default=0.0, type=float)
    
    return parser.parse_args(args)


def set_logger(args):
    log_file = os.path.join(args.save_path, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, epoch, metrics):
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, epoch, metrics[metric]))


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_triple_with_reverse(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    并且加入了逆关系
    '''
    triples = []
    relation_num = len(relation2id)
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
            triples.append((entity2id[t], relation2id[r]+relation_num, entity2id[h]))
    return triples


def read_tripels_with_ids(file_path):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            h, r, t = int(h), int(r), int(t)
            triples.append((h,r,t))
    return triples
    
    
def read_data(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    logging.info('#total entity: %d' % nentity)
    logging.info('#total relation: %d' % nrelation)
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    return train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation


def read_data_reverse(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    nentity = len(entity2id)
    nrelation = len(relation2id)*2
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    train_triples = read_triple_with_reverse(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple_with_reverse(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple_with_reverse(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    logging.info('#total entity: %d' % nentity)
    logging.info('#total relation: %d' % nrelation)
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    return train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation