###################################### IMPORTS

# Basic
import argparse
import os
import shutil
import time
import sys
import math
import numpy as np
import pathlib.Path
import cv2

# Pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
#import torch.nn.functional as F
#from torch.utils.data import DataLoader

# Torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


###################################### ARGUMENTS

parser = argparse.ArgumentParser(description='Specify some hyper parametres:')

# Training
parser.add_argument("-b",   help="Batch size",             default=64,     type=int)
parser.add_argument("-lr",  help="Learning rate",          default=0.01,   type=float)
parser.add_argument("-mo",  help="Momentum",               default=0.9,    type=float)
parser.add_argument("-wd",  help="Weight decay",           default=0.0005, type=float)
parser.add_argument("-e",   help="Number of total epochs", default=5,      type=int)
parser.add_argument("-dlb", help="Drop last batch",        action="store_true")
parser.add_argument('-r',   help='path to latest checkpoint' default='',   type=str,   metavar='PATH')

# Hardware
parser.add_argument("-cpu", help="Do not use cuda",        action="store_true")
parser.add_argument('-gpus',help='Use multiple GPUs',      action='store_true')

# Data
parser.add_argument('data', metavar='DIR', help='path to dataset') # [imagenet-folder with train and val folders]

parser.add_argument("-vp",  help="Validation percentage",  default=0.05,  type=float)
parser.add_argument('-nw',  help="Number of workers",      default=4,     type=int)

# Model
parser.add_argument('-a',   help='Model architecture: ',   default='resnet18', type=str)
parser.add_argument("-pre", help="Load pre-trained model", action="store_true")
parser.add_argument("-cp",  help="Use check_point",        action="store_true")


# Debug
parser.add_argument('-e', '--evaluate',        dest='evaluate',   action='store_true',  help='evaluate model on validation set')
parser.add_argument('--print-freq', '-p',      default=10,   type=int,   metavar='N',   help='print frequency (default: 10)')
parser.add_argument('-v',   help="Pring debug info",       action="store_true")
parser.add_argument('-d',   help="Remote debug",           action="store_true")

args = parser.parse_args()


epochs          = args.e
batch_size      = args.b
learning_rate   = args.lr
momentum        = args.mo
weight_decay    = args.wd
resume          = args.r

val_percent     = args.vp
gpu             = not args.cpu
multiple_gpus   = args.gpus
num_workers     = args.nw
drop_last_batch = args.dlb

architecture    = args.a
pretrained      = args.pre
check_point     = args.cp
verbose         = args.v

# TODO: Poner como argumentos
gpu_count       = 1
images_per_gpu  = 1

#shuffle_data    = True
CROP_SIZE       = 256


###################################### DATA

'''
In the "data" directoy:
 - The "kaggle" dir has the original data from Kaggle.
 - The "masks" dir will be automatically generated with the merged masks.
   Delete "masks" dir if you want to generate the merged masks again (ex. new data).
'''

DATA_DIR          = Path("data")
KAGGLE_TEST_DIR   = DATA_DIR / "kaggle" / "stage1_test"
KAGGLE_TRAIN_DIR  = DATA_DIR / "kaggle-dsbowl-2018-dataset-fixes" / "stage1_train"
KAGGLE_TRAIN_CSV  = DATA_DIR / "kaggle-dsbowl-2018-dataset-fixes" / "stage1_train_labels.csv"
MASKS_DIR         = DATA_DIR / "masks"


dataset    = nucleiDataset(KAGGLE_TRAIN_DIR, MASKS_DIR, CROP_SIZE, cfg.PIXEL_MEANS)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last_batch)
# TODO: Validation dataloader

train_size = len(dataset)
if drop_last_batch: number_of_batches = math.floor(train_size/batch_size)
else:               number_of_batches = math.ceil(train_size/batch_size)


###################################### DATA NEW


traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)



###################################### MODEL

# Select a torchvision model
if pretrained:
    print("=> using pre-trained model '{}'".format(architecture))
    model = torchvision.models.__dict__[architecture](pretrained=True)
else:
    print("=> creating model '{}'".format(architecture))
    model = torchvision.models.__dict__[architecture]()


if architecture.startswith('alexnet') or architecture.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()


#if multiple_gpus:
#    fasterRCNN = nn.DataParallel(fasterRCNN) # Prepare net to support multiple gpus
#
#if gpu:
#    fasterRCNN.cuda() # Load net to gpu (or gpus)



###################################### LOSS function (criterion) and optimizer

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)

# optionally resume from a checkpoint
if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

cudnn.benchmark = True




###################################### TRAIN NEW

def train(train_loader, model, criterion, optimizer, epoch):
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate_epoch(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # FOWARD: compute output and loss
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # BACKWARD: compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate_epoch(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # FOWARD: compute output and loss
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
