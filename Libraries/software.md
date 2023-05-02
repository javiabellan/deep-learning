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

## From pip


```bash
pip install --user jupyterthemes     # Themes for Jupyter Notebooks (jt -t onedork -T)
pip install --user kaggle            # Kaggle API
pip install --user fastai            # Deep learning and machine learning library (PyTorch)
pip install --user pytorch-ignite    # High-level library to help training (PyTorch)
pip install --user pyro-ppl          # Deep probabilistic programming library (PyTorch)
pip install --user Keras             # Deep Learning for humans (Tensorflow)
pip install --user torchvision       # Image and video datasets and models for torch (PyTorch)
pip install --user torchtext         # [NLP] Data loaders and abstractions for text and NLP (PyTorch)
pip install --user nltk              # [NLP]
pip install --user spacy             # [NLP] Industrial-Strength Natural Language Processing
pip install --user allennlp          # [NLP] An Apache 2.0 NLP research library (PyTorch)

pip install --user bcolz
pip install --user graphviz          # Interface for python
pip install --user sklearn-pandas
pip install --user pandas-summary    # An extension to pandas describe function.
pip install --user isoweek

pip install git+https://github.com/fastai/fastai                      # Latest fastai code
pip install git+https://github.com/facebookresearch/fastText.git      # [NLP] Facebook word vectors
```

> If there library is only in github
>
> ```bash
> $ git clone https://github.com/facebookresearch/fastText.git
> $ cd fastText
> $ pip install .    or     $ pip install . -e
> ```
> or
> ```bash
> pip install git+https://github.com/facebookresearch/fastText.git
> ```
