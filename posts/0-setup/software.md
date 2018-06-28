# Software

## Operating System

I tried install DL software on MacOS, Debian and Ubuntu.
And it was a pain to get all tha job done and keep updated my software:
* Compile the source code sometimes.
* Not having a central packager manager to keep updated all my software

Until I found Arch linux and its package manager called `pacman` that get done all the job for me.
Here is an example to install the most imporntant software for DL:

```bash
pacman -S nvidia                     # Nvidia drivers
pacman -S cuda                       # Nvidia API for GPGPU
pacman -S cudnn                      # Nvidia CUDA Deep Neural Network library
pacman -S python                     # Python 3
pacman -S jupyter-notebook           # Python notebooks
pacman -S python-numpy               # Matrix manipulation
pacman -S python-scipy               # Scientific library
pacman -S python-pandas              # Deal with data
pacman -S python-scikit-learn        # Machine learning
pacman -S python-matplotlib          # Visualization
pacman -S python-pytorch-cuda        # Pytorch
pacman -S python-tensorflow-opt-cuda # Tensorflow
pacman -S tensorboard                # Tensorboard
pacman -S opencv                     # Computer Vision Library
```

And to update your system just type `pacman -Syyu`
