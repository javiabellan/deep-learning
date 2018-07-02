<p align="center">
<img width="50%" src="/img/title.png" />
</p>

Here are my personal notes of the  [**deeplearning.ai**](https://www.deeplearning.ai)
and [**fast.ai**](http://www.fast.ai) courses,
and the [**deep learning book**](http://www.deeplearningbook.org/)
which gave me a strong understanding of cutting edge deep learning.
I've write this articles for kepping track my learning path,
but you can use it as a guide for learning (or improving) your DL knowledge.


## 0. Setting up
* [Hardware](/posts/0-setup/hardware.md)
* [Software](/posts/0-setup/software.md)
* [Jupyter Notebooks](/posts/0-setup/jupyter.md)
* [Kaggle](/posts/0-setup/kaggle.md)

## 1. Know the basics
* Chain rule
* [Gradient descent](/posts/gradient_descent.md) (training loop)
  * Choose waight initalization (random,...)
  * Choose learining rate (constant, )
* [Activation functions](/posts/activation_functions.md)
  * **ReLU**: Non-linearity compontent of the net (hidden layers)
  * **Softmax**: Sigle-label classification (last layer)
  * **Sigmoid**: Multi-label classification (last layer)
  * **Hyperbolic tangent**:
* [Loss functions](/posts/loss_functions.md) (Criterium)
  * **Mean Absolute Error** (L1 loss): Regression (for bounding boxes?).
  * **Mean Squared Error** (L2 loss): Regression. Penalice bad misses by too much (for single continuous value?).
  * **Cross Entropy**: Sigle-label classification. Usually with **softmax**.
    * **Negative Log Likelihood** is the one-hot simplified version, see [this](https://jamesmccaffrey.wordpress.com/2016/09/25/log-loss-and-cross-entropy-are-almost-the-same/)
  * **Binary Cross Entropy**:  Multi-label classification. Usually with **sigmoid**.

## 2. Start training
* [Find learning rate](/posts/learning_rate.md)
* [Set batch size](/posts/batch_size.md)
  * Batch gradient descent
  * Stochastic gradient descent
  * Mini-batch gradient descent. The biggger the better (but slower). Usually `64` or `128`
* [Normalize inputs](/posts/input_normalization.md): Scale the inputs to have mean 0 and a variance of 1. Also linear decorrelation/whitening/pca helps a lot.
  * `(x-x.min()) / (x.max()-x.min())`: Values from 0 to 1
  * `2*(x-x.min()) / (x.max()-x.min()) - 1`: Values from -1 to 1
  * `(x-x.mean()) / x.std()`: Values from ? to ?, but mean at 0 (most used)
* [Set a good validation set](/posts/validation_set.md)

## 3. Avoid overfitting (try in that order)
1. Get more data
2. [Data augmentation](/posts/data_augmentation.md)
   * [Test time augmentation?](/posts/TTA.md)
3. Generalizable architectures?: add more bachnorm layers, more densenets...
4. Regularization
   * [Dropout](/posts/dropout.md). Usually `0.5`
   * [Weight decay](/posts/weight_decay.md) (Regularization in loss function) (penalice high weights)
     * L1 regularization: add the sum of the absolute wights. `l1_loss = loss + 0.0005(sum(abs(w)))`
     * L2 regularization: add the sum of the squared wights. `l2_loss = loss + 0.0005(sum(sqrt(w)))`
5. Reduce model complexity

## 4. Train faster (Optimization)
* [SGD with restarts](http://ruder.io/deep-learning-optimization-2017)
* [Gradient Descent Optimization](http://ruder.io/optimizing-gradient-descent)
  * **Momentum**. Usually `0.9` `torch.optim.SGD( momentum=0.9)` The second most used. 
  * **AdaGrad** (Adaptative lr) `torch.optim.Adagrad()`
  * **RMSProp** (Adaptative lr) `torch.optim.RMSprop()`
  * **Adam** (Momentun + RMSProp) `torch.optim.Adam()` The **best** and most used. 
* [Weight initialization](/posts/weight_inilatization.md)
  * Random
  * Other better than random?
  * Pretrainded models (transfer learning) **best**
    1. Replace last layer
    2. Fine-tune new layers
    3. Fine-tune more layers (optional)
* [Batch Normalization](/posts/batch-normalization.md)

## 5. Computer vision
* [Convolutional Neural Network (CNN)](/posts/vision/cnn.md) For fixed size oredered data, like images
* [Residual Network (ResNet)](/posts/vision/resnet.md)
* [Siamese network](/posts/vision/siamese.md)
* [Object detection](/posts/vision/detection.md)
* Enhacement:
  * Colorization
  * Super-resolution
  * Artistic style
* [Generative advesarial network (GAN)](/posts/vision/gan.md)
* [Inceptionism](/posts/vision/inceptionism.md)
* [Capsule net](/posts/vision/capsule.md)

## 6. Natural Language Processing
* [Word embedding](/teoría/modelos/embedding.md)
* [Recurrent Neural network (RNN)](/teoría/modelos/rnn.md) For sequences that need keeping the state, like text
* [Translation](/teoría/nlp/Translation.md)
* [Sequence to sequence](/teoría/nlp/seq2seq.md)
* [Attention](/teoría/nlp/attention.md)
* [Large vocabularies](/teoría/nlp/large-vocabs.md)

## 7. Sturctured data
* Continuous variables: Feed them directly to the network
* Categorical variable: Use embeddings


## 8. Large datasets
* Large images
* Lots of data points
* Large outputs

## 9. How to face a DL problem
1. Think what is your best
   * Data
   * Architecture
   * Loss function
2. Start with a terrible model that overfit a lot

## 10. Advices
* Experiment a lot, especially in your aera of expertise.
* Get known to the world!
  * Don't wait to be perfect before you start communicating.
  * If you don't have a blog, try [Medium](https://medium.com/)

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
  * DQN
* Policy gradients
  * A3C
* Evolutionary Strategy
* Genetic Algorithms
