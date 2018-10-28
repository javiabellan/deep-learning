
########################################################## DeepLearner
"""

Welcome to the deeplearner framework

Other Pytorch-based framwirks
    https://github.com/fastai/fastai
    https://github.com/Cadene/bootstrap.pytorch

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

    def __init__(self, dataset, model_name, loss=None, pretrained=True, augmentation=None):

        # Hardware
        self.cpu_cores     = num_cpus()
        self.gpu           = torch.cuda.is_available() # check for better performance: torch.backends.cudnn.enabled
        self.gpus          = torch.cuda.device_count() > 1

        # Data
        self.train_dataset = dataset[0]
        self.valid_dataset = dataset[1]
        self.val_percent   = 0.2

        self.batch_size    = 64
        self.num_workers   = self.cpu_cores
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers, pin_memory=True)
        self.valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        self.train_size    = len(self.train_dataset)
        self.num_batches   = math.ceil(train_size/batch_size) if not drop_last_batch else math.floor(train_size/batch_size)


        # Model
        self.model_name    = model_name
        self.model         = vision_model(self.model_name, pretrained=pretrained)
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


    def vision_model(self, model_name, pretrained=True):
        # super(TwoLayerNet, self).__init__()
        # super().__init__(x_transform, y_transform)

        try:
            model_fn = eval("torchvision.models."+model_name) # Create model fn
            model    = model_fn(pretrained=pretrained)        # Create model
            
            if self.gpus: model = nn.DataParallel(model)      # Prepare model to support multiple gpus
            if self.gpu:  model.cuda()                        # Load model to gpu (or gpus)
            
            return model

        except AttributeError:
            print(model_name+" model don't exists in torchvision")


    def change_last_layer(self, num_outputs):
        self.freeze()
        num_features  = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_outputs) # New last layer is unfreezed


    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
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

        if resume: self.load_chechpoint(filename)
        for epoch in range(epochs):
            self.adjust_learning_rate(optimizer, epoch)                  # adjust learning rate
            self.train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
            self.validate(val_loader, model, criterion)                  # evaluate on validation set
            self.save_chechpoint(epoch)                                  # Save model parameters

    # train one epoch
    def train(train_loader, model, criterion, optimizer, epoch):
        model.train() # switch to train mode
        self.init_stats()

        for i, (input, target) in enumerate(train_loader):

            # GET DATA
            target = target.cuda(non_blocking=True)

            # FOWARD: compute output and loss
            output = model(input)
            loss = criterion(output, target)

            # BACKWARD: compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # STATISTICS: loss and accuracy (avg)
            self.batch_stats(samples_seen, output, target, loss.item(), start_batch_time)


    # validate one epoch
    def validate(val_loader, model, criterion):
        model.eval() # switch to evaluate mode
        self.init_stats()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)

                # FOWARD: compute output and loss
                output = model(input)
                loss = criterion(output, target)

                # STATISTICS: loss and accuracy (avg)
                self.batch_stats(samples_seen, output, target, loss.item(), start_batch_time)


    ########################################################## TRAIN STATISTICS


    def init_stats():
        self.samples_batch, self.samples_seen = 0, 0
        self.sum_loss, self.avg_loss          = 0, 0
        self.sum_top1, self.avg_top1          = 0, 0
        self.sum_top5, self.avg_top5          = 0, 0
        self.start_epoch_time                 = time.time()
        self.start_batch_time                 = start_epoch_time


    def batch_stats(n, output, target, loss=None, start_epoch_time)

        batch_time   = time.time() - start_batch_time         # Get batch duration
        prec1, prec5 = accuracy(output, target, topk=(1, 5))  # Get top1 and top5 accuracy

        samples_batch =  input.size(0)
        samples_seen  += samples_batch

        sum_loss += loss.item() * samples_batch
        avg_loss = sum_loss / samples_seen

        sum_top1 += prec1 * samples_batch
        avg_top1 = sum_top1 / samples_seen

        sum_top5 += prec5 * samples_batch
        avg_top5 = sum_top5 / samples_seen

        if i % self.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        start_batch_time = time.time()                       # Reset batch duration


    def epoch_stats():
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

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


    ########################################################## TRAIN CHECKPOINTS

    def save_chechpoint(self, epoch, save_all=False, save_last=True, save_best=True):

        filename=epoch+"-"+self.model_name+'checkpoint.pth.tar'

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        {
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict()
        }

        torch.save(state, filename)

        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def load_chechpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))

            checkpoint  = torch.load(filename)
            start_epoch = checkpoint['epoch']
            best_prec1  = checkpoint['best_prec1']
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))


    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr



    def get_data(data, drop_last_batch=False):

        data_dir  = pathlib.Path(data)
        train_dir = data_dir / "train"
        valid_dir = data_dir / "valid"
        test_dir  = data_dir / "test"