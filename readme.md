<p align="center">
<img width="50%" src="/img/title.png" />
</p>

Here are my personal notes of the  [**deeplearning.ai**](https://www.deeplearning.ai)
and [**fast.ai**](http://www.fast.ai) courses,
and the [**deep learning book**](http://www.deeplearningbook.org/)
which gave me a strong understanding of cutting edge deep learning.
I've write this articles for kepping track my learning path,
but you can use it as a guide for learning (or improving) your DL knowledge.

> TODO:
> [check this pytorch DL course](https://fleuret.org/ee559/)
> [trask book](https://github.com/iamtrask/Grokking-Deep-Learning)

## 0. Setting up
* [Hardware](/posts/0-setup/hardware.md)
* [Software](/posts/0-setup/software.md)
* [Jupyter Notebooks](/posts/0-setup/jupyter.md)
* [Kaggle](/posts/0-setup/kaggle.md)

## 1. Know the basics
* Data
  * Training: used for learning the parameters of the model. 
  * Validation: used for evaluating model while training
  * Test data: used to get a final estimate of how well the network works.
* [Chain rule](/posts/1-basics/chain_rule.md)
* [Gradient descent](/posts/1-basics/gradient_descent.md) (training loop)
  * Choose waight initalization (random,...)
  * Choose learining rate (constant, )
* [Activation functions](/posts/1-basics/activations.md)
  * **ReLU**: Non-linearity compontent of the net (hidden layers)
  * **Softmax**: Sigle-label classification (last layer)
  * **Sigmoid**: Multi-label classification (last layer)
  * **Hyperbolic tangent**:
* [Loss functions](/posts/1-basics/loss.md) (Criterium)
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
  * N-fold cross-validation. Usually `10`
* Train until start overffiting (early stopping)

> TODO: Read:
> * [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (1998, Yann LeCun)
> * LR finder
>   * [blog](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
>   * [paper](https://arxiv.org/abs/1506.01186)
> * [Superconvergence](https://arxiv.org/abs/1708.07120)
> * [A disciplined approach to neural network hyper-parameters](https://arxiv.org/pdf/1803.09820.pdf) (2018, Leslie Smith)
> * [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)




## 3. Improve generalization and avoid overfitting (try in that order)
1. Get more data
2. [Data augmentation](/posts/3-generalization/data_augmentation.md)
   * [Test time augmentation (TTA)?](/posts/3-generalization/TTA.md)
3. Generalizable architectures?: add more bachnorm layers, more densenets...
4. Regularization
   * [Dropout](/posts/3-generalization/dropout.md). Usually `0.5`
   * [Weight penalty](/posts/3-generalization/weight_decay.md): Regularization in loss function (penalice high weights). Usually `0.0005`
     * L1 regularization: penalizes the sum of absolute weights.
     * L2 regularization: penalizes the sum of squared weights (weight decay).
5. Reduce model complexity: Limit the number of hidden layers and the number of units per layer.
6. Other approaches
   * **Ensemble**: Average many different models with different forms.
   * **Bagging**: Train the model on different subsets of the training data.
   * **Bayesian**: Use a single architecture, but average the predictions made by many different weight vectors.  

## 4. Train faster (Optimization)
* [SGD with restarts](/posts/4-optimization/sgd-with-restarts.md)
* [Gradient Descent Optimization](/posts/4-optimization/sgd-optimization.md)
  * **Momentum**. Usually `0.9` The second most used. 
  * **AdaGrad** (Adaptative lr)
  * **RMSProp** (Adaptative lr)
  * **Adam** (Momentun + RMSProp) The **best** and most used. 
* [Weight initialization](/posts/4-optimization/weight_inilatization.md)
  * Random
  * Other better than random?
  * Pretrainded models (transfer learning) **best**
    1. Replace last layer
    2. Fine-tune new layers
    3. Fine-tune more layers (optional)
* [Batch Normalization](/posts/4-optimization/batch-normalization.md)
* [2nd order optimization](/posts/4-optimization/2nd-order.md)

## 5. Computer vision
* [Convolutional Neural Network (CNN)](/posts/5-vision/cnn.md) For fixed size oredered data, like images
* [Residual Network (ResNet)](/posts/5-vision/resnet.md)
* [Object detection](/posts/5-vision/detection.md)
  * Class Activation Maps (CAM)
  * Single-object detection
  * Multi-object detection (SSD)
* Enhacement:
  * Colorization
  * Super-resolution
  * Artistic style
* [Generative advesarial network (GAN)](/posts/5-vision/gan.md)
  * [infoGAN](http://www.depthfirstlearning.com/2018/InfoGAN)
  * [10 types of GANs](https://amp.reddit.com/r/MachineLearning/comments/8z97mx/r_math_insights_from_10_gan_papers_infogans)
* [Siamese network](/posts/5-vision/siamese.md)
* [Inceptionism](/posts/5-vision/inceptionism.md)
* [Capsule net](/posts/5-vision/capsule.md)

## 6. Natural Language Processing
* [Word embedding](/teoría/modelos/embedding.md)
* [Recurrent Neural network (RNN)](/teoría/modelos/rnn.md) For sequences that need keeping the state, like text
  * LSTM
  * Neural Turing machine
    * [paper](https://arxiv.org/abs/1807.08518)
    * [code](https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/README.md)
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
