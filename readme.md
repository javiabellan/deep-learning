<p align="center"><img width="50%" src="posts/img/title.png" /></p>

Here are my personal notes of the  [**deeplearning.ai**](https://www.deeplearning.ai)
and [**fast.ai**](http://www.fast.ai) courses,
and the [**deep learning book**](http://www.deeplearningbook.org/)
which gave me a strong understanding of cutting edge deep learning.
I've write this articles for kepping track my learning path,
but you can use it as a guide for learning (or improving) your DL knowledge.

> TODO:
> - [How to train your resnet](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/)
> - [DL cheatsheets](https://stanford.edu/~shervine/teaching/cs-230.html)
> - [Pytorch DL course](https://fleuret.org/ee559/)
> - [Trask book](https://github.com/iamtrask/Grokking-Deep-Learning)

## 0. Setting up
- [Hardware](/posts/0-setup/hardware.md)
- [Software](/posts/0-setup/software.md)
- [Jupyter Notebooks](/posts/0-setup/jupyter.md)
- [Kaggle](/posts/0-setup/kaggle.md)

## 1. Know the basics
> Remember the math: [Matrix calculus](http://explained.ai/matrix-calculus/index.html)

- [Chain rule](/posts/1-basics/chain_rule.md)
- [Gradient descent](/posts/1-basics/gradient_descent.md) (training loop)
  - Choose waight initalization (random,...)
  - Choose learining rate (constant, )
- Batch size: Number of samples to learn simultaneously. Usually a power of 2. `32` or `64` are good values.
  - Too low: like `4`: Lot of updates. Very noisy random updates in the net (bad).
  - Too high: like `512` Few updates. Very general common updates (bad).
    - Faster computation. Takes advantage of GPU mem. But sometimes it can no be fitted (CUDA Out Of Memory)
- [Activation functions](/posts/1-basics/activations.md)
  - **ReLU**: Non-linearity compontent of the net (hidden layers)
  - **Softmax**: Sigle-label classification (last layer)
  - **Sigmoid**: Multi-label classification (last layer)
  - **Hyperbolic tangent**:
- [Loss functions](/posts/1-basics/loss.md) (Criterium)
  - **Mean Absolute Error** (L1 loss): Regression (for bounding boxes?).
  - **Mean Squared Error** (L2 loss): Regression. Penalice bad misses by too much (for single continuous value?).
  - **Cross Entropy**: Sigle-label classification. Usually with **softmax**.
    - **Negative Log Likelihood** is the one-hot simplified version, see [this](https://jamesmccaffrey.wordpress.com/2016/09/25/log-loss-and-cross-entropy-are-almost-the-same/)
  - **Binary Cross Entropy**:  Multi-label classification. Usually with **sigmoid**.
- **Classification Metrics**: Dataset with 5 disease images and 20 normal images. If the model predicts all images to be normal, its accuracy is 80%, and F1-score of such a model is 0.88
  - **Accuracy**: `TP + TN / TP + TN + FP + FN`
  - **F1 Score**: `2 * (Prec*Rec)/(Prec+Rec)`
    - **Precision**: `TP / TP + FP` = `TP / predicted possitives`
    - **Recall**: `TP / TP + FN` = `TP / actual possitives`
  - **Dice Score**: `2 * (Pred ∩ GT)/(Pred + GT)`
  - **ROC, AUC**:
  - **Log loss**:
- **Regression Metrics**
  - - **RMSE**

## Prepare de data
- **Balance** the data
  - **Fix it in the dataloader** [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)
  - **Subsample majority class**. But you can lose important data.
  - **Oversample minority class**. But you can overfit.
  - **Weighted loss function** `CrossEntropyLoss(weight=[…])`
- **Split** the data
  - **Training** set: used for learning the parameters of the model. 
  - **Validation** set: used for evaluating model while training. Very important!
    - N-fold cross-validation. Usually `10`
  - **Test** set: used to get a final estimate of how well the network works.
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
  
## 2. Start training
- [Find learning rate](/posts/learning_rate.md)
- [Set batch size](/posts/batch_size.md)
  - Batch gradient descent
  - Stochastic gradient descent
  - Mini-batch gradient descent. The biggger the better (but slower). Usually `64` or `128`
- Train until start overffiting (early stopping)

> TODO: Read:
> - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (1998, Yann LeCun)
> - LR finder
>   - [blog](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
>   - [paper](https://arxiv.org/abs/1506.01186)
> - [Superconvergence](https://arxiv.org/abs/1708.07120)
> - [A disciplined approach to neural network hyper-parameters](https://arxiv.org/pdf/1803.09820.pdf) (2018, Leslie Smith)
> - [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)


## 3. Improve generalization and avoid overfitting (try in that order)
1. Get more data
2. [Data augmentation](/posts/3-generalization/data_augmentation.md) ([albumentations](https://github.com/albu/albumentations) for faster aug. using the GPU)
   - [Test time augmentation (TTA)?](/posts/3-generalization/TTA.md)
3. Generalizable architectures?: add more bachnorm layers, more densenets...
4. Regularization
   - [Dropout](/posts/3-generalization/dropout.md). Usually `0.5`
   - [Weight penalty](/posts/3-generalization/weight_decay.md): Regularization in loss function (penalice high weights). Usually `0.0005`
     - L1 regularization: penalizes the sum of absolute weights.
     - L2 regularization: penalizes the sum of squared weights (weight decay).
5. Reduce model complexity: Limit the number of hidden layers and the number of units per layer.
6. Ensambles
   - **Bagging** (ensembling): Combine few models and average the predicction.
   - **Stacking** (meta ensembling): Same but use a new model to produce the final output.
   - **Snapshot Ensembling**: M models for the cost of 1. Thanks to SGD with restarts you have several local minimum that you can average. [paper](https://arxiv.org/abs/1704.00109).

## 4. Train faster (Optimization)
- [SGD with restarts](/posts/4-optimization/sgd-with-restarts.md)
- [Gradient Descent Optimization](/posts/4-optimization/sgd-optimization.md)
  - **Momentum**. Usually `0.9` The second most used. 
  - **AdaGrad** (Adaptative lr)
  - **RMSProp** (Adaptative lr)
  - **Adam** (Momentun + RMSProp) The **best** and most used. 
- [Weight initialization](/posts/4-optimization/weight_inilatization.md)
  - Random
  - Other better than random?
  - Pretrainded models (transfer learning) **best**
    1. Replace last layer
    2. Fine-tune new layers
    3. Fine-tune more layers (optional)
- **Precomputation**
  1. Freeze the layers you don’t want to modify
  2. Calculate the activations the last layer from the frozen layers(for your entire dataset)
  3. Save those activations to disk
  4. Use those activations as the input of your trainable layers
- [**Half precision**](https://forums.fast.ai/t/mixed-precision-training/20720) (fp16)
- **Multiple GPUs**
- [**Batch Normalization**](/posts/4-optimization/batch-normalization.md)
- [**2nd order optimization**](/posts/4-optimization/2nd-order.md)


> [Normalization inside network](https://nealjean.com/ml/neural-network-normalization):
> - Batch Normalization [paper](https://arxiv.org/abs/1502.03167)
> - Layer Normalization [paper](https://arxiv.org/abs/1607.06450)
> - Instance Normalization [paper](https://arxiv.org/abs/1607.08022)
> - Group Normalization [paper](https://arxiv.org/abs/1803.08494)

## 5. Computer vision
> TODO: Read [inception nets](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
- [Convolutional Neural Network (CNN)](/posts/5-vision/cnn.md) For fixed size oredered data, like images
  - Variable input size: use **adaptative pooling**, final layers then:
    - Option 1: `AdaptiveAvgPool2d((1, 1))` -> `Linear(num_features, num_classes)` (less computation)
    - Option 2: `Conv2d(num_features, num_classes, 3, padding=1)` -> `AdaptiveAvgPool2d((1, 1))`
- [CNN Black box explanation (for classification)](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
  - Class Activation Maps (CAM)
  - SmoothGrad ([paper](https://arxiv.org/abs/1706.03825))
  - Extra: [Distill: feature visualization](https://distill.pub/2017/feature-visualization/)
  - Extra: [Distill: building blocks](https://distill.pub/2018/building-blocks/)
- [Object detection](/posts/5-vision/detection.md)
  - Single-object detection
  - Multi-object detection (SSD)
- Segmentation
  - [Unet](https://github.com/facebookresearch/fastMRI/tree/master/models/unet)
- Enhacement:
  - Colorization
  - Super-resolution
  - Artistic style
- [Generative advesarial network (GAN)](/posts/5-vision/gan.md)
  - [infoGAN](http://www.depthfirstlearning.com/2018/InfoGAN)
  - [10 types of GANs](https://amp.reddit.com/r/MachineLearning/comments/8z97mx/r_math_insights_from_10_gan_papers_infogans)
- [Siamese network](/posts/5-vision/siamese.md)
- [Inceptionism](/posts/5-vision/inceptionism.md)
- [Capsule net](/posts/5-vision/capsule.md)

> To speed up jpeg image I/O from the disk one should not use PIL, skimage and even OpenCV but look for libjpeg-turbo or PyVips.

## 6. Natural Language Processing
- [Word embedding](/teoría/modelos/embedding.md)
- [Recurrent Neural network (RNN)](/teoría/modelos/rnn.md) For sequences that need keeping the state, like text
  - LSTM
  - Neural Turing machine
    - [paper](https://arxiv.org/abs/1807.08518)
    - [code](https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/README.md)
- [Translation](/teoría/nlp/Translation.md)
- [Sequence to sequence](/teoría/nlp/seq2seq.md)
- [Attention](/teoría/nlp/attention.md)
- [Large vocabularies](/teoría/nlp/large-vocabs.md)

## 7. Sturctured data
- Continuous variables: Feed them directly to the network
- Categorical variable: Use embeddings
- Collaborative filtering: How much a user is going to like a certain item

---

## Unsupervised learning
* [Autoencoder](/teoría/modelos/autoencoder.md): Para comprimir información
* Restricted boltzmann machine: Como el autoencoder pero va y vuelve
* PCA: Reduce el numero de dimensiones
* T-SNE: Reduce el numero de dimensiones
* competitive learning
* Hebbian learning
* Self Organizing Map

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

## Machine learning
* [Gradient boosting](http://explained.ai/gradient-boosting/index.html)
* Random forest
