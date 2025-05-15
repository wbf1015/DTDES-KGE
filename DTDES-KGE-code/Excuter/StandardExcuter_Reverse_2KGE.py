import sys
import os
import math
import logging
import shutil
import torch.nn.functional as F
import torch
import torch.nn as nn
import tqdm

CODEPATH = os.path.abspath(os.path.dirname(__file__))
CODEPATH = CODEPATH.rsplit('/', 1)[0]
sys.path.append(CODEPATH)

from Optim.Optim import *

class StandardExcuter_Reverse_2KGE(object):
    def __init__(self, 
                 KGE1=None,
                 KGE2=None,
                 model=None,
                 embedding_manager=None, 
                 entity_pruner=None, relation_pruner=None, 
                 decoder=None,
                 trainDataloader=None, testDataLoaders=None,
                 args=None,
    ):
        self.args = args
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        self.model = model(KGE1=KGE1, KGE2=KGE2, embedding_manager=embedding_manager, entity_pruner=entity_pruner, relation_pruner=relation_pruner, decoder=decoder,  args=args)

        if KGE1.__class__.__name__ == 'RotatE_Reverse':
            pass
        if KGE1.__class__.__name__ == 'HAKE_Reverse':
            pass
        if KGE1.__class__.__name__ == 'AttH_Reverse':
            self.load_KGE1_AttH(self.model, args)
        if KGE1.__class__.__name__ == 'HyperNet':
            self.load_KGE1_LorentzE(self.model, args)
        if KGE1.__class__.__name__ == 'MRME_KGC':
            self.load_KGE1_MRME(self.model, args)
        
        if KGE2.__class__.__name__ == 'RotatE_Reverse':
            pass
        if KGE2.__class__.__name__ == 'HAKE_Reverse':
            pass
        if KGE2.__class__.__name__ == 'AttH_Reverse':
            self.load_KGE2_AttH(self.model, args)
        if KGE2.__class__.__name__ == 'HyperNet':
            self.load_KGE2_LorentzE(self.model, args)
        if KGE2.__class__.__name__ == 'MRME_KGC':
            self.load_KGE2_MRME(self.model, args)
        
        self.trainDataloader = trainDataloader
        self.testDataLoaders = testDataLoaders
        self.args = args
        
        if self.args.init_checkpoint != 'without':
            self.load_model(self.model, args)
            
        if self.args.cuda:
            self.model.cuda()
        
        if self.args.data_type == 'double':
            for param in self.model.parameters():
                param.data = param.data.double()
        else:
            for param in self.model.parameters():
                param.data = param.data.float()
        
        
        self.optimizer = getoptimizer(args, filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = getscheduler(args, self.optimizer, last_epoch=-1)
        
    def Run(self):
        
        # metric = self.test_model(1)
        # self.log_metrics(0, metric)
        # sys.exit(0)
        
        training_loss = []
        
        train_data_len = self.trainDataloader.total_length
        
        for epoch in range(1, self.args.epoch+1):
            
            with tqdm.tqdm(total=train_data_len*self.args.batch_size) as bar:
                bar.set_description(f'Epoch {epoch} Train Loss')
                
                for step in range(train_data_len):
                    
                    loss = self.train_step(epoch=epoch)
                    training_loss.append(loss)
                    bar.update(self.args.batch_size)
                    bar.set_postfix(loss=loss['loss'])
                    
            self.save_model(self.model, self.args)
            
            def calculate_metrics(logs, prefix=''):
                metrics = {}
                # Initialize a dictionary to keep track of the count for each metric
                metric_counts = {}

                # Iterate through each log entry
                for log in logs:
                    for metric, value in log.items():
                        # Initialize metric sum and count if it's the first occurrence
                        if metric not in metrics:
                            metrics[metric] = 0.0
                            metric_counts[metric] = 0
                        # Add the value and increment the count for this metric
                        metrics[metric] += value
                        metric_counts[metric] += 1

                # Compute the average for each metric where count is non-zero
                for metric in metrics.keys():
                    if metric_counts[metric] > 0:
                        metrics[metric] /= metric_counts[metric]
                    else:
                        metrics[metric] = None  # Handle cases with no valid logs for a metric

                # Log the metrics with the provided prefix
                self.log_metrics(epoch, metrics)
            
            calculate_metrics(training_loss)
            training_loss = []
            
            # metric = self.test_model(1)
            # self.log_metrics(self.args.epoch, metric)
            # self.save_model(self.model, self.args)
            # sys.exit(0)
            
            if (epoch)%self.args.save_checkpoint_epochs==0:
                self.save_model(self.model, self.args)
            
            if ((epoch)%(self.args.test_per_epochs)==0):
                metric = self.test_model(1)
                self.log_metrics(epoch, metric)
                self.save_model(self.model, self.args)
            
        
        # metric = self.test_model(1)
        # self.log_metrics(self.args.epoch, metric)
        # self.save_model(self.model, self.args)
        
        
    def train_step(self, epoch):
        self.optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(self.trainDataloader)
        if self.args.cuda:
            positive_sample, negative_sample, subsampling_weight = positive_sample.cuda(), negative_sample.cuda(), subsampling_weight.cuda()
        loss, loss_record = self.model((positive_sample, negative_sample), subsampling_weight, mode, epoch)

        if torch.isnan(loss).item():
            logging.info('NaN Loss appear, Auto Exit')
            sys.exit(-1)
        
        loss.backward()
        self.optimizer.step()
        
        return loss_record

    def test_model(self, predict_func=1):
        self.model.eval()
        with torch.no_grad():
            logs = []
            step = 0
            total_steps = sum([len(dataset) for dataset in self.testDataLoaders])
            result_record = []
            for test_dataset in self.testDataLoaders:
                for positive_sample, negative_sample, filter_bias in test_dataset:
                    if self.args.cuda:
                        positive_sample, negative_sample, filter_bias = positive_sample.cuda(), negative_sample.cuda(), filter_bias.cuda()
                        
                    batch_size = positive_sample.size(0)
                    score = self.model.predict((positive_sample, negative_sample), predict_func)
                    score = score[:, 1:]
                    
                    # 用于测试教师模型是否正确导入
                    # PT1_score, PT2_score = self.model.get_KGEScore((positive_sample, negative_sample))
                    # score = PT1_score[:, 1:]
                    # score = PT2_score[:, 1:]
                    
                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    if self.args.add_bias:
                        score += filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=True) # 降序
                    else:
                        score -= filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=False)

                    positive_arg = positive_sample[:, 2]

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            'HITS@50': 1.0 if ranking <= 50 else 0.0,
                            'HITS@100': 1.0 if ranking <= 100 else 0.0,
                        })

                        result_record.append(f"{positive_sample[i, 0].item()}\t{positive_sample[i, 1].item()}\t{positive_sample[i, 2].item()}\t{ranking}")
                        
                    if step % self.args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        # 这段代码的作用是保存学生模型对于所有测试集的排名
        output_path = os.path.join(self.args.save_path, f'test_detail_result_{predict_func}.txt')
        with open(output_path, 'w') as f:
            for record in result_record:
                f.write(record + "\n")
        
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        self.model.train()
        return metrics
    
    
    def log_metrics(self, epoch, metrics):
        for metric in metrics:
            logging.info('%s at epoch %d: %f' % (metric, epoch, metrics[metric]))
    
    
    def write_dict_to_txt(self, dictionary, file_path):
        with open(file_path, 'w') as file:
            for key, value in dictionary.items():
                file.write(f"{key}: {value}\n")
    
    
    def save_model(self, model, args):
        argparse_dict = vars(args)
        self.write_dict_to_txt(argparse_dict, os.path.join(args.save_path, 'config.json'))

        Runpy_path = CODEPATH + '/Run.py'
        myrunsh_path = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 2)[0] + '/myrun.sh'
        runsh_path = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 2)[0] + '/run.sh'
        
        # 文件列表和它们的新名称
        files_to_copy = [Runpy_path, myrunsh_path, runsh_path]
        new_names = [os.path.join(args.save_path, os.path.basename(f) + 'Store' + os.path.splitext(f)[1]) for f in files_to_copy]

        # 复制并重命名文件
        for original, new in zip(files_to_copy, new_names):
            shutil.copy2(original, new)  # 使用 copy2 以保留元数据
        
        # 把模型和embedding的值都存上
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     },
        #     os.path.join(args.save_path, 'checkpoint')
        # )
    
    def load_model(self, model, args):
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    def load_KGE1_AttH(self, model, args):
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))
        
        model.KGE1.c = torch.nn.Parameter(pretrain_model['model_state_dict']['KGE.c'].cpu().to(self.data_type))
        model.KGE1.bh.weight.data = pretrain_model['model_state_dict']['KGE.bh.weight'].cpu().to(self.data_type)
        model.KGE1.bt.weight.data = pretrain_model['model_state_dict']['KGE.bt.weight'].cpu().to(self.data_type)       
        model.KGE1.rel_diag.weight.data = pretrain_model['model_state_dict']['KGE.rel_diag.weight'].cpu().to(self.data_type)
        model.KGE1.context_vec.weight.data = pretrain_model['model_state_dict']['KGE.context_vec.weight'].cpu().to(self.data_type)
        
        # 固定所有参数的 requires_grad 为 False
        for param in model.KGE1.parameters():
            param.requires_grad = False   
    
    def load_KGE2_AttH(self, model, args):
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'))
        
        model.KGE2.c = torch.nn.Parameter(pretrain_model['model_state_dict']['KGE.c'].cpu().to(self.data_type))
        model.KGE2.bh.weight.data = pretrain_model['model_state_dict']['KGE.bh.weight'].cpu().to(self.data_type)
        model.KGE2.bt.weight.data = pretrain_model['model_state_dict']['KGE.bt.weight'].cpu().to(self.data_type)       
        model.KGE2.rel_diag.weight.data = pretrain_model['model_state_dict']['KGE.rel_diag.weight'].cpu().to(self.data_type)
        model.KGE2.context_vec.weight.data = pretrain_model['model_state_dict']['KGE.context_vec.weight'].cpu().to(self.data_type)
        
        # 固定所有参数的 requires_grad 为 False
        for param in model.KGE2.parameters():
            param.requires_grad = False
    
    def load_KGE1_LorentzE(self, model, args):
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'), map_location='cpu')
        model.KGE1.load_state_dict(pretrain_model['model_state_dict'])
        
        for param in model.KGE1.parameters():
            param.data = param.data.to(dtype=self.data_type)
        
        # 固定所有参数的 requires_grad 为 False
        for param in model.KGE1.parameters():
            param.requires_grad = False    
    
    
    def load_KGE2_LorentzE(self, model, args):
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'), map_location='cpu')
        model.KGE2.load_state_dict(pretrain_model['model_state_dict'])
        
        for param in model.KGE2.parameters():
            param.data = param.data.to(dtype=self.data_type)
        # model.KGE2 = model.KGE2.to(dtype=self.data_type)
        
        # 固定所有参数的 requires_grad 为 False
        for param in model.KGE2.parameters():
            param.requires_grad = False

    def load_KGE1_MRME(self, model, args):
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'), map_location='cpu')
        model.KGE1.load_state_dict(pretrain_model)
        
        for param in model.KGE1.parameters():
            param.data = param.data.to(dtype=self.data_type)
        # model.KGE2 = model.KGE2.to(dtype=self.data_type)
        
        # 固定所有参数的 requires_grad 为 False
        for param in model.KGE1.parameters():
            param.requires_grad = False 
    
    def load_KGE2_MRME(self, model, args):
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'), map_location='cpu')
        model.KGE2.load_state_dict(pretrain_model)
        
        for param in model.KGE2.parameters():
            param.data = param.data.to(dtype=self.data_type)
        
        # 固定所有参数的 requires_grad 为 False
        for param in model.KGE2.parameters():
            param.requires_grad = False