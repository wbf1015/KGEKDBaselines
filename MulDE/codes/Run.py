import torch
import numpy as np

from DataLoader.RandomSample import *
from EmbeddingManager.MulDEManager import *
from KGES.TransE import *
from KGES.RotatE import *
from Models.MulDEModel import *
from Loss.MulDELoss import *
from Optim.Optim import *
from Excuter.MulDEExcuter import *
from utils import *

args = parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
set_logger(args)

np.random.randn(args.seed)
torch.manual_seed(args.seed)


'''
声明数据集
'''
train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation = read_data(args)

train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)
train_dataloader_tail = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)
test_dataloader_head = DataLoader(
    TestDataset(
        test_triples, 
        all_true_triples, 
        args.nentity, 
        args.nrelation, 
        'head-batch'
    ), 
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2), 
    collate_fn=TestDataset.collate_fn
)
test_dataloader_tail = DataLoader(
    TestDataset(
        test_triples, 
        all_true_triples, 
        args.nentity, 
        args.nrelation, 
        'tail-batch'
    ), 
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2), 
    collate_fn=TestDataset.collate_fn
)
logging.info('Successfully init TrainDataLoader and TestDataLoader')

'''
声明Excuter组件
'''
KGE=TransE(margin=args.gamma)
# KGE=RotatE(margin=args.gamma, embedding_range=args.gamma+2.0, embedding_dim=args.target_dim)

muldeloss = MulDELoss(adv_temperature = args.adversarial_temperature, margin = args.gamma, l=args.soft_loss_weight)
embedding_manager=MulDEManager(args)

trainDataloader =BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
testDataLoaders=[test_dataloader_head, test_dataloader_tail]

optimizer=None
scheduler=None

Excuter = MulDEExcuter(
    KGE=KGE, 
    model=MulDEModel,
    embedding_manager=embedding_manager, 
    muldeloss=muldeloss,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    optimizer=optimizer, scheduler=scheduler,
    args=args,
    ntriples = len(train_dataloader_head),
)

Excuter.Run()
