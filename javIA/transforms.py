from torchvision.transforms import *



# Resize
rand_crop224 = [RandomResizedCrop(224)]
cent_crop224 = [CenterCrop(224)]
resize256    = [Resize(256)]


# Augmentation
aug_basic    = [ColorJitter(0.1, 0.1, 0.1), RandomRotation(10)] # Basic augmentation: Light and rotation
aug_side_on  = aug_basic   + [RandomHorizontalFlip()]           # Extra augmentation: For normal photos
aug_top_down = aug_side_on + [RandomVerticalFlip()]             # Extra augmentation: For satellite and medical imgs


# To tensor and normalize
mean         = [0.485, 0.456, 0.406]
std          = [0.229, 0.224, 0.225]
final        = [ToTensor(), Normalize(mean=mean, std=std)]



train_tfms   = Compose(rand_crop224 + aug_side_on + final)
valid_tfms   = Compose(resize256 + cent_crop224 + final)

