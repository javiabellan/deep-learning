import os
import torch
import pandas as pd
# from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision      import transforms



HOME_DIR   = "/home/javi/DL/kaggle/whales"
DATA_DIR   = f"{HOME_DIR}/data"
TEST_DIR   = f"{DATA_DIR}/test"
TRAIN_DIR  = f"{DATA_DIR}/train"
TRAIN_CSV  = f"{DATA_DIR}/train.csv"
CROPS_CSV  = f"{PATH}/crops.csv"


class WhaleDataset(Dataset):

	def __init__(self, csv_file, data_dir, transform=None):
		self.labels    = pd.read_csv(csv_file)
		self.data_dir  = data_dir
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img_name = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
		# image  = skimage.io.imread(img_name)   # Skimage
		image    = Image.open(img_name)
		# image = np.array(image)                # PIL image to numpy array
		
		label    = self.labels.iloc[idx, 1]
		sample   = {"image": image, "label": label}

		if self.transform:
			sample = self.transform(sample)

		return sample


dataset    = WhaleDataset(TRAIN_CSV, TRAIN_DIR)
# dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=nw)


train_size = len(dataset)
