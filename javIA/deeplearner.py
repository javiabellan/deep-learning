
########################################################## DeepLearner
"""

Welcome to the deeplearner framework

TODO:
Change lr (sgdr)
https://github.com/fastai/fastai/blob/master/courses/dl2/training_phase.ipynb


"""
########################################################## IMPORTS

from utils import * 

# Basic
import time
import sys
import pathlib

# Pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
#import torch.nn.functional as F
#from torch.utils.data import DataLoader



class DeepLearner():

    ########################################################## CONSTRUCTOR

    def __init__(self, data_dir="data", model, loss=None, pretrained=True, augmentation=None):

        # Hardware
        self.cpu_cores     = num_cpus()
        self.gpu           = torch.cuda.is_available() # check for better performance: torch.backends.cudnn.enabled
        self.gpus          = torch.cuda.device_count() > 1

        # Data
        self.get_data(data_dir)
        self.train_dataset = self.get_dataset(train_dir, transforms)
        self.test_dataset  = self.get_dataset(train_dir, transforms)

        # Model
        self.model         = self.get_torchvision_model(model, pretrained)
        self.criterion     = self.get_loss(loss)

        # Training
        self.epochs        = 1
        self.learning_rate = 0.01
        self.momentum      = 0.9
        self.weight_decay  = 0.0005
        self.resume        = False
        self.optimizer     = self.get_optimizer()

        # This is good whenever your input sizes for your network do not vary.
        cudnn.benchmark = True


    ########################################################## DATA

    def get_transforms(type):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if type=="train":
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ])
        elif type=="valid":
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize
            ])

        return transforms

    def get_data(data, drop_last_batch=False):

        data_dir  = pathlib.Path(data)
        train_dir = data_dir / "train"
        valid_dir = data_dir / "valid"
        # test_dir  = data_dir / "test"


        self.val_percent   = 0.2
        self.train_dataset = torchvision.datasets.ImageFolder(train_dir, get_transforms("train")) # Or custom dataset
        self.valid_dataset = torchvision.datasets.ImageFolder(valid_dir, get_transforms("valid")) # Or custom dataset

        self.batch_size    = 64
        self.num_workers   = self.cpu_cores
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers, pin_memory=True, drop_last=drop_last_batch)
        self.valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=drop_last_batch)

        self.train_size    = len(self.train_dataset)
        self.num_batches   = math.ceil(train_size/batch_size) if not drop_last_batch else math.floor(train_size/batch_size)


    ########################################################## MODEL

    def get_torchvision_model(model_name, pretrained=True):

        try:
            model_fn = eval("torchvision.models."+model_name) # Create model fn
            model = model_fn(pretrained=pretrained)           # Create model

            if self.gpus: model = nn.DataParallel(model)      # Prepare model to support multiple gpus
            if self.gpu:  model.cuda()                        # Load model to gpu (or gpus)

            return model

        except AttributeError:
            print(model_name+" model don't exists in torchvision")


    def change_last_layer(num_outputs):
        self.freeze()
        num_features  = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_outputs) # New last layer is unfreezed


    def freeze():
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze():
        for param in self.model.parameters():
            param.requires_grad = True




    ########################################################## LOSS

    def get_loss(self, classification=True, single_label=True):

        if classification and single_label:
            loss =  torch.nn.CrossEntropyLoss() # CE  = Softmax + NLLLoss

        elif classification and not single_label:
            loss = torch.nn.BCEWithLogitsLoss() # BCE = Sigmoid + BCELoss

        else: # regression
            loss = torch.nn.MSELoss()

        # if self.gpu: loss.cuda()

        return loss



    def get_optimizer(self):
        return torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)


    ########################################################## TRAIN

    def fit(self, epochs, lr, n_cycle, wd=None, resume=False):
        if resume: self.resume()
        for epoch in range(epochs):

            self.adjust_learning_rate(optimizer, epoch)              # adjust learning rate
            train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
            prec1 = validate(val_loader, model, criterion)           # evaluate on validation set

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict()}, is_best)

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    # optionally resume from a checkpoint
    def resume(self):
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))

            checkpoint  = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1  = checkpoint['best_prec1']
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))


    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


    # train one epoch
    def train(train_loader, model, criterion, optimizer, epoch):
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

    # validate one epoch
    def validate(val_loader, model, criterion):
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
