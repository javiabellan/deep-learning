from datasets import *
from transforms import *
from deeplearner import *

train_dir = "/home/javi/DL/fastai/courses/dl1/data/dogscats/train"
valid_dir = "/home/javi/DL/fastai/courses/dl1/data/dogscats/valid"

train_dataset = ImageFolderDataset(train_dir, train_tfms)
valid_dataset = ImageFolderDataset(valid_dir, valid_tfms)


dl = deeplearner(model="resnet34", data="/home/javi/DL/fastai/courses/dl1/data/dogscats")
dl.change_last_layer(2)
dl.summary()
dl.lr_find()
dl.fit(epochs=2, lr=xxx, cycle_len=1, wd=1e-5)
dl.unfreeze()


dl.fit(epochs=2, batch_size=32, lr=xxx, cycle_len=1)


