import torch
import numpy as np

from DataLoader.QueryAwareSample_Reverse import *
from EmbeddingManager.KDManager_Reverse_2KGE import *
from KGES.RotatE_Reverse import *
from KGES.HAKE_Reverse import *
from KGES.AttH_Reverse import *
from KGES.SCCF_Reverse import *
from KGES.LorentzKG_Reverse.LorentzKG_Reverse import *
from KGES.MRME_Reverse.MRME_Reverse import *
from Models.EncoderModel_Reverse_2KGE import *
from Optim.Optim import *
from Decoder.Logits_Feature_Decoder.LFDecoder import *
from Extracter.Constant import *
from Excuter.StandardExcuter_Reverse_2KGE import *
from utils import *

args = parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
set_logger(args)


'''
声明数据集
'''
train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation = read_data_reverse(args)

train_dataloader = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, args=args), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)

test_dataloader = DataLoader(
    TestDataset(
        test_triples, 
        all_true_triples, 
        args.nentity, 
        args.nrelation, 
    ), 
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2), 
    collate_fn=TestDataset.collate_fn
)
logging.info('Successfully init TrainDataLoader and TestDataLoader')


'''
声明Excuter组件
'''

if 'FB15k-237' in args.data_path:
    KGE1 = RotatE_Reverse(teacher_margin=2.0, teacher_embedding_dim=256) # RotatE its self
    KGE2 = RotatE_Reverse(teacher_margin=9.0, teacher_embedding_dim=256) # LorentzKGEpoch27-RotatE-FB15k-237-256
    # KGE2 = MRME_KGC(sizes=(args.nentity, args.nrelation, args.nentity), rank=100) # MRME
elif 'wn18rr' in args.data_path:
    KGE1 = RotatE_Reverse(teacher_margin=2.0, teacher_embedding_dim=256) # RotatE its self
    KGE2 = RotatE_Reverse(teacher_margin=4.0, teacher_embedding_dim=256)  # LorentzKGEpoch5-RotatE-FB15k-237-256
    # KGE2 = MRME_KGC(sizes=(args.nentity, args.nrelation, args.nentity), rank=150) # MRME
elif 'YAGO3-10' in args.data_path:
    KGE1 = HAKE_Reverse(args=args, teacher_embedding_dim=500) # HAKE
    KGE2 = MRME_KGC(sizes=(args.nentity, args.nrelation, args.nentity), rank=50) # MRME


entity_pruner=Constant()
relation_pruner=Constant()


decoder = Decoder_2KGE(args)
embedding_manager=KDManager_Reverse_2KGE(args)


# trainDataloader =SimpleBidirectionalOneShotIterator(train_dataloader)
trainDataloader = BidirectionalOneShotIterator(train_dataloader)
testDataLoaders=[test_dataloader]


Excuter = StandardExcuter_Reverse_2KGE(
    KGE1=KGE1,
    KGE2=KGE2, 
    model=EncoderModel_Reverse_2KGE,
    embedding_manager=embedding_manager, 
    entity_pruner=entity_pruner, relation_pruner=relation_pruner, 
    decoder=decoder,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    args=args
)

Excuter.Run()
