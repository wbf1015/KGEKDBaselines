import torch
import numpy as np

from DataLoader.RandomSample import *
from EmbeddingManager.IterDEManager import *
from KGES.TransE import *
from KGES.RotatE import *
from Models.IterDEModel import *
from Loss.SigmoidLossOrigin import *
from Loss.HuberLoss import *
from Optim.Optim import *
from Excuter.IterDEExcuter import *
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
# KGE=TransE(margin=None)
KGE=RotatE(margin=None, embedding_range=args.gamma+2.0, embedding_dim=args.target_dim)

if args.negative_adversarial_sampling is False:
    hard_loss=SigmoidLossOrigin(adv_temperature=None, margin=args.gamma)
else:
    hard_loss=SigmoidLossOrigin(adv_temperature=args.adversarial_temperature, margin=args.gamma)

soft_loss = HuberLoss(args.soft_loss_weight)

embedding_manager=IterDEManager(args)
trainDataloader =BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
testDataLoaders=[test_dataloader_head, test_dataloader_tail]

optimizer=None
scheduler=None

Excuter = IterDEExcuter(
    KGE=KGE, 
    model=IterDEModel,
    embedding_manager=embedding_manager, 
    soft_loss=soft_loss,
    hard_loss=hard_loss,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    optimizer=optimizer, scheduler=scheduler,
    args=args,
)

Excuter.Run()
