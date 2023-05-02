#https://linuxtut.com/en/be92c8fdaeaec9b506e4/

import numpy as np
import cv2
import sys
import albumentations as A


#time djpeg man.jpg | convert - -flop - | convert - -resize 100x100 - > /dev/null
#djpeg man.jpg                            0,00s user 0,00s system  51% cpu 0,008 total
#convert - -flop -                        0,04s user 0,01s system 209% cpu 0,023 total
#convert - -resize 100x100 - > /dev/null  0,02s user 0,01s system  63% cpu 0,048 total

############################### Bo (/landmark-recognition-2020) 3rd palce

image_size = 224
train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),               # Targets: image, mask, bboxes, keypoints
        A.ImageCompression(quality_lower=99,   # Targets: image
                           quality_upper=100), 
        A.ShiftScaleRotate(shift_limit=0.2,    # Targets: image, mask, keypoints
                           scale_limit=0.2,
                           rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT_101,
                           p=0.7),
        A.Resize(image_size, image_size),     # Targets: image, mask, bboxes, keypoints
        #A.Cutout(max_h_size=int(image_size * 0.4),
        #        max_w_size=int(image_size * 0.4),
        #        num_holes=1,
        #        p=0.5),
        A.CoarseDropout(max_height=int(image_size * 0.4), #Targets: image, mask, keypoints
                 max_width=int(image_size * 0.4),
                 max_holes=1,
                 min_holes=1,
                 fill_value=0,
                 mask_fill_value=0,
                 p=0.5),
        # A.Normalize()   # Targets: image
    ])

val_aug = A.Compose([
        A.Resize(image_size, image_size),
    ])


############################### MAIN

# READ FROM STD_IN
path = sys.argv[1]
img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# TRANSFORM
aug = train_aug(image=img[..., :3], mask=img[...,3])
aug_rgba = np.dstack((aug["image"], aug["mask"]))

# WRITE TO STD_OUT
flag, buf = cv2.imencode('.png', aug_rgba)
sys.stdout.buffer.write(buf.tobytes())
