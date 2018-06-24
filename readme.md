<p align="center">
<img width="50%" src="https://cdn-images-1.medium.com/max/1200/1*VcdHE40-TqZ3anN-YHk5uQ.png" />
</p>

Here are my personal notes of the  [**deeplearning.ai**](https://www.deeplearning.ai)
and [**fast.ai**](http://www.fast.ai) courses,
and the [**deep learning book**](http://www.deeplearningbook.org/)
which gave me a strong understanding of cutting edge deep learning.
I've write this articles for kepping track my learning path,
but you can use it as a guide for learning (or improving) your DL knowledge.


## 0. Setting up
* [Hardware and Operating System](/posts/hardware_and_os.md)
* [Python](/posts/python.md)
* [Jupyter notebooks](/posts/jupyter.md)
* [Pytorch](/posts/pytorch.md)
* [Kaggle](/posts/kaggle.md)

## 1. Know the basiscs
* Chain rule
* [Gradient descent](/posts/gradient_descent.md) (training loop)
  * Choose waight initalization (random,...)
  * Choose learining rate (constant, )
* [Activation functions](/posts/activation_functions.md)
  * **ReLU**: Non-linearity compontent of the net (hidden layers)
  * **Softmax**: Sigle-label classification (last layer)
  * **Sigmoid**: Milti-label classification (last layer)
* [Loss functions](/posts/loss_functions.md)

## 2. Start training
* [Find learning rate](/posts/learning_rate.md)
* [Set batch size](/posts/batch_size.md)
  * Batch gradient descent
  * Stochastic gradient descent
  * Mini-batch gradient descent. The biggger the better (but slower). Usually `64` or `128`
* [Normalize inputs](/posts/input_normalization.md)
* [Weight initialization](/posts/weight_initialization.md)
* [Set a good validation set](/posts/validation_set.md)

## 3. Fight overfitting (Regularization)
* [Dropout](/posts/dropout.md). Usually `0.5`
* [Data augmentation](/posts/data_augmentation.md)
* [Test time augmentation](/posts/TTA.md)
* [Weight decay](/posts/weight_decay.md) (Regularization in loss function) (penalice high weights)
  * L1 regularization: add the sum of the absolute wights. `l1_loss = loss + 0.0005(sum(abs(w)))`
  * L2 regularization: add the sum of the squared wights. `l2_loss = loss + 0.0005(sum(sqrt(w)))`

## 4. Train faster (Optimization)
* [SGD with restarts](http://ruder.io/deep-learning-optimization-2017)
* [Gradient Descent Optimization](http://ruder.io/optimizing-gradient-descent)
  * Momentum. Usually `0.9`
  * AdaGrad (Adaptative lr)
  * RMSProp (Adaptative lr)
  * Adam (Momentun + RMSProp) The best and most used.
* [Weight inilatization](/posts/weight_inilatization.md) (better than random)
* Batch Normalization
* Pretrainded models (transfer learning)

## 5. Computer vision
* [Redes neuronales convolucionales](/teoría/modelos/cnn.md)
* [Localización de objetos](/teoría/modelos/cnn.md)
* [Inceptionism](/teoría/modelos/Inceptionism.md)
* [Capsule net](/teoría/modelos/capsule.md)

## 6. Natural Language Processing
* [Word embedding](/teoría/modelos/embedding.md)
* [Red neuronal recurrente](/teoría/modelos/rnn.md)
* [Sequence to sequence](/teoría/modelos/seq2seq.md)

## 7. Sturctured data

## Audio
* Procesar audio

#### Generative models
* Generative advesarial network (GAN)

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
