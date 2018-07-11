"""
TODO:

Augmentations in openCV:
https://github.com/albu/albumentations
https://github.com/aleju/imgaug

"""

from torchvision.transforms import *


size           = 224 # For all nets: resnet, etc...
incep_size     = 299 # For inception


# 1. Resize
rand_crop    = [RandomResizedCrop(size)]
cent_crop    = [CenterCrop(size)]
resize256    = [Resize(256)]


# 2. Augmentation
aug_basic    = [ColorJitter(0.1, 0.1, 0.1), RandomRotation(10)] # Basic augmentation: Light and rotation
aug_side_on  = aug_basic   + [RandomHorizontalFlip()]           # Extra augmentation: For normal photos
aug_top_down = aug_side_on + [RandomVerticalFlip()]             # Extra augmentation: For satellite and medical imgs


# 3. To tensor and normalize
mean         = [0.485, 0.456, 0.406]
std          = [0.229, 0.224, 0.225]
incep_mean   = [0.5, 0.5, 0.5]
incep_std    = [0.5, 0.5, 0.5]
final        = [ToTensor(), Normalize(mean=mean, std=std)]



train_tfms   = Compose(rand_crop + aug_side_on + final)
valid_tfms   = Compose(resize256 + cent_crop + final)

