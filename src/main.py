import argparse
from xml.dom import ValidationErr
import torch
import torch.nn as nn
import torch.optim as optim

from model import NetworkTSC
from trainer import Trainer
from utils import load_data, data_loader

from torch.optim.lr_scheduler import MultiStepLR

def parse_args():
    parser = argparse.ArgumentParser(description='Traffic Sign Classification...')
    #directory
    parser.add_argument('--dataroot',     type=str,   default="/mnt/d/dataset/traffic_sign/", help='path to dataset')
    parser.add_argument('--ckptroot',     type=str,   default="../checkpoints/",          help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr',           type=float, default=1e-4,          help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,          help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size',   type=int,   default=32,            help='training batch size')
    parser.add_argument('--num_workers',  type=int,   default=8,             help='# of workers used in dataloader')
    parser.add_argument('--train_size',   type=float, default=0.8,           help='train validation set split ratio')
    parser.add_argument('--shuffle',      type=bool,  default=True,          help='whether shuffle data during training')

    # training settings
    parser.add_argument('--epochs',       type=int,   default=15,            help='number of epochs to train')
    parser.add_argument('--start_epoch',  type=int,   default=0,             help='pre-trained epochs')
    parser.add_argument('--num_classes',       type=int,   default=43,            help='number of epochs to train')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    epoch = args.epochs
    shuffle = args.shuffle

    print("==> Preparing dataset ...")
    #lad data and split to trainset and validset
    trainset, valset = load_data(args.dataroot, args.train_size)
    trainloader, validationloader = data_loader(trainset,valset,args.batch_size,args.shuffle)
    
    #define model
    print("==> Initialize model ...")
    model = NetworkTSC(args.num_classes)
 

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    #cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==> Use accelerator: ", device)
    print("==> start training ...")

    trainer = Trainer(args.epochs, model, optimizer, criterion, trainloader,validationloader,scheduler,args.ckptroot)
    trainer.perform_training()

if __name__=='__main__':
    main()