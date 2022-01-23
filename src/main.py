import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from model import NetworkTSC
from trainer import Trainer
from utils import load_data, data_loader, load_test_data, test_data_loader
from PIL import Image
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import MultiStepLR

# from xml.dom import ValidationErr
# import torchvision
# import torch.utils.data as data
# import torchvision.transforms as transforms


# test_transforms = transforms.Compose([
#     transforms.Resize([112, 112]),
#     transforms.ToTensor()
#     ])


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
    #parser.add_argument('--start_epoch',  type=int,   default=0,             help='pre-trained epochs')
    parser.add_argument('--num_classes',       type=int,   default=43,            help='number of epochs to train')
    parser.add_argument('--start_epoch',  type=int,   default=0,             help='pre-trained epochs')

    #test settings
    parser.add_argument('--test', type=bool,  default=False,        help='test the dataset(True/False)') 
    parser.add_argument('--modelroot',	type=str, default="../checkpoints/model-15.h5", help='name/path of the model')
    parser.add_argument('--test_img',	type=str, default="/mnt/d/dataset/traffic_sign/Train/0/00000_00000_00029.png", help='name/path of the model')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    #define model
    print("==> Initialize model ...")
    model = NetworkTSC(args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    if(args.test):
        
        print("==> Loading pre-trained model...")
        checkpoint = torch.load(args.modelroot)
        print("==> Loading checkpoint model successfully ...")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        print("==> Preparing dataset ...")

        num = range(args.num_classes)
        labels = []
        for i in num:
            labels.append(str(i))
        labels = sorted(labels)
        for i in num:
            labels[i] = int(labels[i])
        # print("list of labels : ")
        # print("Actual labels \t--> Class in PyTorch")
        # for i in num:
        #     print("\t%d \t-->\t%d" % (labels[i], i))
        
        df = pd.read_csv(args.dataroot+"tt/Test.csv")
        numExamples = len(df)
        labels_list = list(df.ClassId)

        testset = load_test_data(args.dataroot)
        testloader = test_data_loader(testset)
        
        # testset = torchvision.datasets.ImageFolder(root = args.dataroot+"Test", transform = test_transforms)
        # testloader = data.DataLoader(testset, batch_size=1, shuffle=False)
        
        output_list = []
        corr_classified = 0

        with torch.no_grad():
            model.eval()
            i=0
            for image, _ in testloader:
                image = image.cuda()
                output_raw = model(image)
                output_softmax = torch.log_softmax(output_raw[0], dim=1)
                _, output_max = torch.max(output_softmax, dim=1)
                output_max = output_max.cpu().numpy()
                output = output_max[0]
                output = labels[output]
                output_list.append(output)

                if labels_list[i] == output:
                    corr_classified +=1
                i +=1

        # print(output_list)
        print("Number of correctly classified images = %d" % corr_classified)
        print("Number of incorrectly classified images = %d" % (numExamples - corr_classified))
        print("Final accuracy = %f" % (corr_classified / numExamples))
        print(classification_report(labels_list, output_list))
                        
    else:
        #cuda or cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", device)
        print("==> Preparing dataset ...")
        #load data and split to trainset and validset
        trainset, valset = load_data(args.dataroot, args.train_size)

        print(f"==> Number of training samples = {len(trainset)}")
        print(f"==> Number of validation samples = {len(valset)}")

        trainloader, validationloader = data_loader(trainset,valset,args.batch_size,args.shuffle)
        print("==> start training ...")

        trainer = Trainer(args.epochs, model, optimizer, criterion, trainloader,validationloader,scheduler,args.ckptroot)
        trainer.perform_training()


if __name__=='__main__':
    main()