# Basic
import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path

# Pytorch
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Local files
from data import *
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.net_utils import clip_gradient
from model.utils.visualization import *
from iterm import show_image

from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv

import cv2



###################################### ARGUMENTS

parser = argparse.ArgumentParser(description='Specify some hyper parametres:')

# Training
parser.add_argument("-b",   help="Batch size",             default=1,      type=int)
parser.add_argument("-l",   help="Learning rate",          default=0.001,  type=float)
parser.add_argument("-mo",  help="Momentum",               default=0.9,    type=float)
parser.add_argument("-wd",  help="Weight decay",           default=0.0005, type=float)
parser.add_argument("-e",   help="Number of max epochs",   default=5,      type=int)
parser.add_argument("-dlb", help="Drop last batch",        action="store_true")

# Hardware
parser.add_argument("-cpu", help="Do not use cuda",        action="store_true")
parser.add_argument('-gpus',help='Use multiple GPUs',      action='store_true')

# Data
parser.add_argument("-vp",  help="Validation percentage",  default=0.05,  type=float)
parser.add_argument('-nw',  help="Number of workers",      default=4,     type=int)

# Model
parser.add_argument("-pt",  help="Load pre-trained model", action="store_true")
parser.add_argument("-cp",  help="Use check_point",        action="store_true")

# Debug
parser.add_argument('-v',   help="Pring debug info",       action="store_true")
parser.add_argument('-d',   help="Remote debug",           action="store_true")

args = parser.parse_args()

if args.d:
    import pydevd
    pydevd.settrace('192.168.0.178', port=55555, stdoutToServer=True, stderrToServer=True)


epochs          = args.e
batch_size      = args.b
learning_rate   = args.l
momentum        = args.mo
weight_decay    = args.wd

val_percent     = args.vp
gpu             = not args.cpu
multiple_gpus   = args.gpus
num_workers     = args.nw
drop_last_batch = args.dlb
check_point     = args.cp
trained_model   = args.pt
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



###################################### MODEL

fasterRCNN = vgg16(classes=[1], pretrained=trained_model, class_agnostic=True) # fasterRCNN with vgg16 as feature extraction
fasterRCNN.create_architecture()

if multiple_gpus:
    fasterRCNN = nn.DataParallel(fasterRCNN) # Prepare net to support multiple gpus

if gpu:
    fasterRCNN.cuda() # Load net to gpu (or gpus)


###################################### TRAIN

params = []
for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
        if 'bias' in key:
            params += [{'params':[value],'lr':learning_rate*2}]
        else:
            params += [{'params':[value],'lr':learning_rate, 'weight_decay': weight_decay}]
    
optimizer = torch.optim.SGD(params, momentum=momentum)
#optimizer = torch.optim.SGD(fasterRCNN.parameters(),lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

fasterRCNN.train() # setting model to train mode
print("\x1b[2J\x1b[;H", end='')
clean_screen = False
print("========== TRAIN ==========")
for epoch in range(epochs):
    print("Epoch ",epoch, "/",epochs)
    for i, batch in enumerate(dataloader):

        #print("\tBatch ", i, "/",number_of_batches)

        # 0) Batch
        images   = batch['image']     # Shape: [batch_size, 3, crop_size, crop_size]
        gt_boxes = batch['bboxs']     # Shape: [batch_size, num_masks, 5]                     5->(x1, y1, x2, y2, cls)
        #gt_masks = batch['masks']    # Shape: [batch_size, num_masks, crop_size, crop_size]        
        img_info  = batch['img_info'] # Shape: [batch_size, 3] 3-> (shape[1], shape[2], scale)
        num_boxes = batch['num_boxs'] # Shape: [batch_size, 1]

        # Wrap in variables
        images    = Variable(images)
        gt_boxes  = Variable(gt_boxes) # TODO: normalize between 0 and 1
        #gt_masks  = Variable(gt_masks)
        img_info  = Variable(img_info)
        num_boxes = Variable(num_boxes)

        # To GPU
        if gpu:
            images    = images.cuda()
            gt_boxes  = gt_boxes.cuda()
            #gt_masks  = gt_masks.cuda()
            img_info  = img_info.cuda()
            num_boxes = num_boxes.cuda()

        # 1) Forward
        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(images, img_info, gt_boxes, num_boxes)

        # 2) Loss
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
       #    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        #loss = rpn_loss_box.mean() + RCNN_loss_bbox.mean()
        #loss_temp += loss.data[0]

        if not i % 64:

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                       + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)

            pred_boxes = boxes # bbox_transform_inv(boxes, box_deltas, 1)
            #pred_boxes = clip_boxes(pred_boxes, img_info.data, 1)

            #pred_boxes /= batch[1][0][2]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            img  = images[0].data.cpu().numpy().transpose(1, 2, 0)
            img  += cfg.PIXEL_MEANS

            img = np.uint8(img).copy()
            bbxs_gt = gt_boxes[0].data.cpu().numpy()
            bbxs_pr = pred_boxes[0].cpu().numpy()

            #img_gt = draw_bounding_boxes_on_image_array(img, bbxs_gt, thickness=2, use_normalized_coordinates=False)
            #img_pr = draw_bounding_boxes_on_image_array(img, bbxs_pr, thickness=2, use_normalized_coordinates=True)

            for i in range(bbxs_gt.shape[0]):
                bbox = tuple(int(np.round(x * CROP_SIZE)) for x in bbxs_gt[i, :4])
                cv2.rectangle(img, bbox[0:2], bbox[2:4], (204, 0, 0), 2)

            for i in range(bbxs_pr.shape[0]):
                bbox = tuple(int(np.round(x * CROP_SIZE)) for x in bbxs_pr[i, :4])
                cv2.rectangle(img, bbox[0:2], bbox[2:4], (0, 204, 0), 1)

            #img_gt = img_gt.squeeze()
            #img_pr = img.squeeze()

            # Printing
            if clean_screen is False:
                print("\x1b[2J\x1b[;H", end='')
                clean_screen = True
            print("\x1b[H")
            print("Epoch: ", epoch+1," | Batch ", i+1, "/",number_of_batches, " | Loss: ", loss.data[0])
            print("\x1b[s", end='')
            show_image(img, 0, 255.)

            print(bbxs_gt[:5])
            print(bbxs_pr[:5])

            #print("\x1b[u\x1b[40G", end='')
            #show_image(img_gt, 0, 255.)

            #print(bbxs_gt)
            #print(bbxs_pr[:20])

                    # 3) Backward
        optimizer.zero_grad()
        loss.backward()
        if True: #args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
        optimizer.step()

# TODO: Get validation loss

        # TODO: Save model


    # Validation: TODO

# Save model: TODO