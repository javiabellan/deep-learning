<p align="center"><img width="50%" src="posts/img/title.png" /></p>

Here are my personal deep learning notes. I've written this cheatsheet for keep track my knowledge
but you can use it as a guide for learning deep learning aswell.

> ## Index
>
> 0. [**Setting up**](#0-setting-up): Get your machine ready for deep learning.
>
> #### Part 1: General concepts
> 
> 1. [**Know the basics**](#1-know-the-basics)
> 2. [**Prepare the data**](#prepare-the-data)
> 3. [**Train & hyperparams**](#2-choose-training-hyperparams)
> 4. [**Avoid overfitting**](#3-improve-generalization-and-avoid-overfitting-try-in-that-order)
> 5. [**Train faster**](#4-train-faster-optimization)
> 
> #### Part 2: Domain specific concepts
> 
> 6. [**Vision**](#5-computer-vision)
> 7. [**NLP**](#6-natural-language-processing)
> 8. [**Table data**](#7-sturctured-data)
> 9. [**Audio**](#8-audio)
> 10. [**Reinforcement learning**](#reinforcement-learning)

---

## 0. Setting up
- [Hardware](/posts/0-setup/hardware.md)
- [Software](/posts/0-setup/software.md)
- [Jupyter Notebooks](/posts/0-setup/jupyter.md)
- [Kaggle](/posts/0-setup/kaggle.md)

## 1. Know the basics
> Remember the math:
> - [Matrix calculus](http://explained.ai/matrix-calculus/index.html)
> - **Einsum**: [link 1](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/), [link 2](https://rockt.github.io/2018/04/30/einsum)

- [Chain rule](/posts/1-basics/chain_rule.md)
- [Gradient descent](/posts/1-basics/gradient_descent.md) (training loop)
  - **Batch** gradient descent: The whole dataset at once, as a batch. `Batch size = length(dataset)`
  - **Online** gradient descent: Every single sample of data is a batch. `Batch size = 1`
  - **Mini-batch** gradient descent: Disjoint groups of samples as a batch. `Batch size = n` **We will use this**.
- [Activation functions](/posts/1-basics/activations.md)
  - **ReLU**: Non-linearity compontent of the net (hidden layers)
  - **Softmax**: Sigle-label classification (last layer)
  - **Sigmoid**: Multi-label classification (last layer)
  - **Hyperbolic tangent**:
- [Loss functions](/posts/1-basics/loss.md) (Criterium)
  - **Regression**
    - **MAE: Mean Absolute Error** (L1 loss): (for bounding boxes?).
    - **MSE: Mean Squared Error** (L2 loss): Penalice large errors more than MAE (for single continuous value?).
    - **RMSE: Root Mean Squared Error**: The square root of the MSE. Penalice large errors more than MAE.
  - **Classification**
    - **Cross Entropy**: Sigle-label classification. Usually with **softmax**. `nn.CrossEntropyLoss`.
    - **NLL: Negative Log Likelihood** is the one-hot simplified version, see [this](https://jamesmccaffrey.wordpress.com/2016/09/25/log-loss-and-cross-entropy-are-almost-the-same/) `nn.NLLLoss()`
    - **Binary Cross Entropy**:  Multi-label classification. Usually with **sigmoid**. `nn.BCELoss`
  - **Hinge**: `nn.HingeEmbeddingLoss()`
- **Classification Metrics**: Dataset with 5 disease images and 20 normal images. If the model predicts all images to be normal, its accuracy is 80%, and F1-score of such a model is 0.88
  - **Accuracy**: `TP + TN / TP + TN + FP + FN`
  - **F1 Score**: `2 * (Prec*Rec)/(Prec+Rec)`
    - **Precision**: `TP / TP + FP` = `TP / predicted possitives`
    - **Recall**: `TP / TP + FN` = `TP / actual possitives`
  - **Dice Score**: `2 * (Pred ∩ GT)/(Pred + GT)`
  - **ROC, AUC**:
  - **Log loss**:
- **Regression Metrics**
  - **RMSE**
  - **RMSPE**: Root Mean Square Percentage Error. `log = true` in fastai to get percentages differences.

## Prepare the data
- **Balance** the data
  - **Fix it in the dataloader** [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)
  - **Subsample majority class**. But you can lose important data.
  - **Oversample minority class**. But you can overfit.
  - **Weighted loss function** `CrossEntropyLoss(weight=[…])`
- **Split** the data
  - **Training set**: used for learning the parameters of the model. 
  - **Validation set**: used for evaluating model while training. Don’t create a random validation set! Manually create one so that it matches the distribution of your data. Usaully a `10%` or `20%` of your train set.
    - N-fold cross-validation. Usually `10`
  - **Test set**: used to get a final estimate of how well the network works.
- [**Preprocess**](http://cs231n.github.io/neural-networks-2/#datapre): Scale the inputs to have mean 0 and a variance of 1. Also linear decorrelation/whitening/pca helps a lot.
  - Option 1: **Normalization** `x = x-x.mean() / x.std()` *Most used*
     1. **Mean subtraction**: Center the data to zero. `x = x - x.mean()` fights vanishing and exploding gradients
     2. **Standardize**: Put the data on the same scale. `x = x / x.std()` improves convergence speed and accuracy
  - Option 2: **PCA Whitening**
    1. **Mean subtraction**: Center the data in zero. `x = x - x.mean()`
    2. **Decorrelation** or **PCA**: Rotate the data until there is no correlation anymore.
    3. **Whitening**: Put the data on the same scale. `whitened = decorrelated / np.sqrt(eigVals + 1e-5)`
  - Option 3: **ZCA whitening** Zero component analysis (ZCA).
  - Other options not used:
    - `(x-x.min()) / (x.max()-x.min())`: Values from 0 to 1
    - `2*(x-x.min()) / (x.max()-x.min()) - 1`: Values from -1 to 1
  
>  In case of images, the scale is from 0 to 255, so it is not strictly necessary normalize.
  
## 2. Choose training hyperparams
- **Learning rate**
  - Constant: Never use.
  - Reduce it gradually: By steps, by a decay factor, with LR annealing, etc.
  - Warm restarts (SGDWR, AdamWR):
  - 1 cycle: The best. Use LRFinder to know your maximum lr.
- **Batch size**: Number of samples to learn simultaneously. Usually a power of 2. `32` or `64` are good values.
  - Too low: like `4`: Lot of updates. Very noisy random updates in the net (bad).
  - Too high: like `512` Few updates. Very general common updates (bad).
    - Faster computation. Takes advantage of GPU mem. But sometimes it can no be fitted (CUDA Out Of Memory)
- **Number of epochs**
  - Train until start overffiting (validation loss becomes to increase) (early stopping)
- [**Gradient descent method**](/posts/4-optimization/sgd-optimization.md)
  - **SGD**. A bit slowly to get to the optimum. `new_w = w - lr[gradient_w]`
  - **SGD with Momentum**. Speed it up with momentum, usually `mom=0.9`. **The second method most used**.
    - `mom=0.9`, means a `10%` is the normal derivative and a `90%` is the same direction I went last time.
    - `new_w = w - lr[(0.1 * gradient_w)  +  (0.9 * w)]`
    - Other common values are `0.5`, `0.7` and `0.99`.
  - **AdaGrad** (Adaptative lr) From 2011.
  - **RMSProp** (Adaptative lr) From 2012. Similar to momentum but with the gradient squared.
    - `new_w = w - lr * gradient_w / [(0.1 * gradient_w²)  +  (0.9 * w)]`
    - If the gradient in not so volatile, take grater steps. Otherwise, take smaller steps.
  - **Adam** Combination of Momentun with RMSProp. From 2014. The **best** and most used.
  - **AMSGrad** From 2018. Worse than Adam in practice.
- **Weight initialization**??? random, xavier...

> TODO: Read:
> - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (1998, Yann LeCun)
> - LR finder
>   - [blog](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
>   - [paper](https://arxiv.org/abs/1506.01186)
> - [Superconvergence](https://arxiv.org/abs/1708.07120)
> - [A disciplined approach to neural network hyper-parameters](https://arxiv.org/pdf/1803.09820.pdf) (2018, Leslie Smith)
> - [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)


## 3. Improve generalization and avoid overfitting (try in that order)
1. **Get more data**
   - **Similar datasets**: Get a similar dataset for your problem.
   - **Create your own dataset**
     - Segmentation annotation with Polygon-RNN++
   - **Synthetic data**: Virtual objects and scenes instead of real images. Infinite possibilities of lighting, colors, angles...
2. **Data augmentation**: Augment your current data. ([albumentations](https://github.com/albu/albumentations) for faster aug. using the GPU)
   - **Test time augmentation (TTA)**: The same augmentations will also be applied when we are predicting (inference). It can improve our results if we run inference multiple times for each sample and average out the predictions.
   - **AutoAugment**: RL for data augmentation. Trasfer learning NOT THE WEIGHTS but the policies of how to do data augmentation.
3. **Regularization**
   - [Dropout](/posts/3-generalization/dropout.md). Usually `0.5`
   - [Weight penalty](/posts/3-generalization/weight_decay.md): Regularization in loss function (penalice high weights). Usually `0.0005`
     - **L1 regularization**: penalizes the sum of absolute weights.
     - **L2 regularization**: penalizes the sum of squared weights by a factor, usually `0.01` or `0.1`.
     - **Weight decay**: `wd * w`. Sometimes mathematically identical to L2 reg.
4. **Reduce model complexity**: Limit the number of hidden layers and the number of units per layer.
   - Generalizable architectures?: Add more bachnorm layers, more densenets...
5. **Ensambles**: Gather a bunch of models to give a final prediction.
   - **Bagging** (ensembling): Average model preditctions (weighted average, majority vote or normal average)
   - **Stacking** (meta ensembling): Same but use a new model to produce the final output.
   - **Snapshot Ensembling**: M models for the cost of 1. Thanks to SGD with restarts you have several local minimum that you can average. [paper](https://arxiv.org/abs/1704.00109).
   - **Boosting** is an ensemble technique in which the predictors are not made independently, but sequentially.
     - [**Gradient Boosting**](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)

## 4. Train faster (Optimization)
- **Transfer learning**: Use a pretrainded model and retrain with your data.
  1. Replace last layer
  2. Fine-tune new layers
  3. Fine-tune more layers (optional)
- [**Batch Normalization**](/posts/4-optimization/batch-normalization.md) Add BachNorm layers after your convolutions and linear layers for make things easier to your net and train faster.
- **Precomputation**
  1. Freeze the layers you don’t want to modify
  2. Calculate the activations the last layer from the frozen layers(for your entire dataset)
  3. Save those activations to disk
  4. Use those activations as the input of your trainable layers
- [**Half precision**](https://forums.fast.ai/t/mixed-precision-training/20720) (fp16)
- **Multiple GPUs**
- [**2nd order optimization**](/posts/4-optimization/2nd-order.md)


> [Normalization inside network](https://nealjean.com/ml/neural-network-normalization):
> - Batch Normalization [paper](https://arxiv.org/abs/1502.03167)
> - Layer Normalization [paper](https://arxiv.org/abs/1607.06450)
> - Instance Normalization [paper](https://arxiv.org/abs/1607.08022)
> - Group Normalization [paper](https://arxiv.org/abs/1803.08494)

## 5. Computer vision
- [Convolutional Neural Network (CNN)](/posts/5-vision/cnn.md) For fixed size oredered data, like images
  - Variable input size: use **adaptative pooling**, final layers then:
    - Option 1: `AdaptiveAvgPool2d((1, 1))` -> `Linear(num_features, num_classes)` (less computation)
    - Option 2: `Conv2d(num_features, num_classes, 3, padding=1)` -> `AdaptiveAvgPool2d((1, 1))`
  - [Inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
  - **Resent**: Every 2 convolutions **sum** the original input.
  - **Denset**: Every 2 convolutions **concatenate** the original input.
  - SENet: Squeeze and Excitation block: network is allowed to adaptively adjust the weighting of each feature map in the convolution block.
- CNN Black box explanation (for classification) [*link 1*](https://github.com/utkuozbulak/pytorch-cnn-visualizations), [*link 2*](https://ramprs.github.io/2017/01/21/Grad-CAM-Making-Off-the-Shelf-Deep-Models-Transparent-through-Visual-Explanations.html)
  - **Features**: Average features on the channel axis. This shows all classes detected. `[512, 11, 11]-->[11, 11]`.
  - **CAM**: Class Activation Map. Final features multiplied by a single class weights and then averaged. `[512, 11, 11]*[512]-->[11, 11]`. [*paper*](https://arxiv.org/abs/1512.04150).
  - **Grad-CAM**: Final features multiplied by class gradients and the averaged. [*paper*](https://arxiv.org/abs/1610.02391).
  - **SmoothGrad** [*paper*](https://arxiv.org/abs/1706.03825).
  - Extra: [Distill: feature visualization](https://distill.pub/2017/feature-visualization/)
  - Extra: [Distill: building blocks](https://distill.pub/2018/building-blocks/)
- [**Object detection** / **Localization**](/posts/5-vision/detection.md): Get bounding boxes
  - **Faster R-CNN**: Region-based method
  - **SSD**: Single-shot method
- **Semantic segmentation**: Get pixels
  - **FCN** Fully Convolutional Networks (2014)
  - **SegNet** (2015)
  - [**Unet**](https://github.com/facebookresearch/fastMRI/tree/master/models/unet): With the left part is a pretrained resnet-34
  - **DeepLabv3** SotA. Increasing dilatation, increases field-of-view. [paper](https://arxiv.org/abs/1706.05587)
- **Generative** (Image-to-image): Useful for data augmentation, B&W colorization, super-resolution, artistic style...
  - **Model**: Pretrained Unet 
  - **Loss functions**:
     - **Pixel MSE**: Flat the 2D images and compare them with regular MSE.
     - **Discriminator/Critic** The loss function is a binary classification pretrained resnet (real/fake).
     - **Feature losses** or perpetual losses.
  - pix2pixHD
  - COVST: Naively add temporal consistency.
  - [Video-to-Video Synthesis](https://tcwang0509.github.io/vid2vid/)
- [Generative advesarial network (GAN)](/posts/5-vision/gan.md)
  - Process
    1. Train a bit the generator and save generated images. `unet_learner` with pixelMSE loss
    2. Train bit the discriminator with real vs generated images. `create_critic_learner`
    3. Ping-pong train both nets `GANLearner` with 2 losses pixelMSE and discriminator.
  - Discriminative model with Spectral Normalization
  - Loss with adaptive loss
  - Metric accuracy is accuracy_thres_expand
  - [infoGAN](http://www.depthfirstlearning.com/2018/InfoGAN)
  - BigGAN: SotA in image synthesis. Same GAN techiques, much larger scale. Increase model capacity + increase batch size.
  - [10 types of GANs](https://amp.reddit.com/r/MachineLearning/comments/8z97mx/r_math_insights_from_10_gan_papers_infogans)
- [Siamese network](/posts/5-vision/siamese.md)
- [Inceptionism](/posts/5-vision/inceptionism.md)
- [Capsule net](/posts/5-vision/capsule.md)

> To speed up jpeg image I/O from the disk one should not use PIL, skimage and even OpenCV but look for libjpeg-turbo or PyVips.

## 6. Natural Language Processing
> - [NLP overview](https://nlpoverview.com/)
> - [Sebastian Ruder webpage](http://ruder.io/)
> - [Jay Alammar webpage](http://jalammar.github.io/)
> - [Hardvard NLP](http://nlp.seas.harvard.edu/papers/)

- [**Word embedding**](/teoría/modelos/embedding.md): Give meaningful representation to words.
  -  **Word2Vec**, **Glove**: Traditional unsupervised process of embedding, where a word is similar to its surrounding words (Skip-gram model)
  - **CNN-extracted char features**
  - **ELMo**: Context-aware embedding = better representation. Useful for synonyms. Made with bidirectional LSTMs [*paper*](https://arxiv.org/abs/1802.05365), [*site*](https://allennlp.org/elmo).
- [**Recurrent Neural network (RNN)**](/teoría/modelos/rnn.md) For sequences that need keeping the state, like text
  - **GRU**
  - **LSTM**
  - Neural Turing machine. [*paper*](https://arxiv.org/abs/1807.08518), [*code*](https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/README.md)
- [**Sequence to sequence**](/teoría/nlp/seq2seq.md): Encoder-Decoder architecture.
  - **Attention** Allows the network to refer back to the input sequence, instead of forcing it to encode all information into ane fixed-lenght vector. [*paper*](https://arxiv.org/abs/1508.04025), [*blog*](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/), [*attention and memory*](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
  - **ULMFiT**: Regular LSTM with no attention. Introduces the idea of transfer-learning in NLP. [*paper*](https://arxiv.org/abs/1801.06146)
  - **Transformer**: Feedfoward network. Encoder with self-attention, and decoder with attention. [*paper*](https://arxiv.org/abs/1706.03762), [*blog*](https://jalammar.github.io/illustrated-transformer).
  - **OpenAI Transformer**: Same as transformer, but with transfer-learning for ther NLP tasks. First train the decoder for language modelling with unsupervised text, and then train other NLP task. [*paper*](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [*site*](https://blog.openai.com/language-unsupervised/)
  - **BERT** The best performance. [*paper*](https://arxiv.org/abs/1810.04805), [*blog*](http://jalammar.github.io/illustrated-bert), [*blog2*](http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/).
  - **Transformer-XL**: Learning long-term dependencies [*paper*](https://arxiv.org/abs/1901.02860), [*blog*](https://medium.com/dair-ai/a-light-introduction-to-transformer-xl-be5737feb13), [*google blog*](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html).
- Metrics
  - [**BLEU**](https://medium.com/@rtatman/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
- [**Applications**](https://nlpprogress.com):
  - [Translation](https://nlpprogress.com/english/machine_translation.html)

## 7. Sturctured data
- Continuous variables: Feed them directly to the network
- Categorical variable: Use embeddings
- Collaborative filtering: How much a user is going to like a certain item

## 8. Audio
> - [Audio overview](https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89)

## 9. DL resources
> TODO:
> - [**fast.ai**](http://www.fast.ai)
> - [**deeplearning.ai**](https://www.deeplearning.ai)
> - [**deep learning book**](http://www.deeplearningbook.org/)
> - [DL cheatsheets](https://stanford.edu/~shervine/teaching/cs-230.html)
> - [How to train your resnet](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/)
> - [Pytorch DL course](https://fleuret.org/ee559/)
> - [Trask book](https://github.com/iamtrask/Grokking-Deep-Learning)
> - [mlexplained](http://mlexplained.com/)


---


## No neural Machine Learning

* **Supervised learning**: Useful for know a non neural baseline or for use them in ensambles along with nueral nets.
  * [**Gradient boosting**](http://explained.ai/gradient-boosting/index.html)
    * **Gradient boosting trees**
  * **Random forest (RF)**
  * **Support Vector Machines (SVM)**
  * **K nearest neighbors (KNN)**
  * **Linear regression**: Regrssion
* **Clustering**: Separate data in groups, useful for labeling a dataset.
  * **K-Means**
  * **Mean-Shift**
  * **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise
* **Dimensionality reduction**: Useful for visualize data and embeddings
  * **PCA**:
  * **T-SNE**:
* **Others**
  * [Autoencoder](/teoría/modelos/autoencoder.md): Para comprimir información
  * Restricted boltzmann machine: Como el autoencoder pero va y vuelve
  * competitive learning
  * Hebbian learning
  * Self Organizing Map
 
 
## Traditional feature extraction
* Color features
* Texture features

## Reinforcement learning
* Q-learning
  * [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* Policy gradients
  * A3C
* [C51](https://arxiv.org/abs/1707.06887)
* [Rainbow](https://arxiv.org/abs/1710.02298)
* [Implicit Quantile](https://arxiv.org/abs/1806.06923)
* Evolutionary Strategy
* Genetic Algorithms

> TODO: Read:
> * [RL’s foundational flaw](https://thegradient.pub/why-rl-is-flawed/)
> * [How to fix reinforcement learning](https://thegradient.pub/how-to-fix-rl/)
> * [AlphaGoZero](http://www.depthfirstlearning.com/2018/AlphaGoZero)
> * [Trust Region Policy Optimization](http://www.depthfirstlearning.com/2018/TRPO)
> * [Udacity repo](https://github.com/udacity/deep-reinforcement-learning)
> * [Introduction to RL Algorithms. Part I](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)
> * [Introduction to RL Algorithms. Part II](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-part-ii-trpo-ppo-87f2c5919bb9)
> * [pytorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
> * [RL Adventure](https://github.com/higgsfield/RL-Adventure)
> * [RL Adventure 2](https://github.com/higgsfield/RL-Adventure-2)
> * [DeepRL](https://github.com/ShangtongZhang/DeepRL)


## Fast.ai API

- Databunch
  - If the label is an int by default is a classification problem.
  - If the label is an float by default is a regression problem.

