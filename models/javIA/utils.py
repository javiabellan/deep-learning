import os
import math
import numpy as np

import cv2


def floor(x):
	return int(math.floor(x))

def ceil(x):
	return int(math.ceil(x))

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

def one_hot(idx, num_classes):
    return np.eye(num_classes)[idx]

def ones_hot(idxs, num_classes):
    res = np.zeros(num_classes, dtype=np.float32)
    res[idxs] = 1
    return res


def cuda_memory_used():
    used_bytes = torch.cuda.memory_allocated() # memory occupied by tensors in bytes

