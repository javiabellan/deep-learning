# SGD Optimization


To train our neural net we detailed the algorithm of Stochastic [Gradient descent](/posts/1-basics/gradient_descent.md).
To make it easier for our model to learn, there are a few ways we can improve it.

### Gradient Descent

Just to remember what we are talking about,
the basic algorithm consists in changing each of our parameter where:
* `p` represents one of our parameters (weights and bias)
* `p.grad` is the gradient of the loss with respect to p
* `lr` is an hyperparameter called learning rate

```
new_p = p - lr * p.grad
```

### Gradient Descent with Momentum
To accelerate thigs up, one improvement is to use mumentum that sum to the actual step, a little bit of the previous step.

```
             _______________grad_actual_______________
new_w = w + ( - (lr)(derivative) + (p)(grad_anterior) )
```

Momentum `p` is usually `.9`. Other common values are `.5`, `.7`, `.9` and `.99`.

In Pytorch: `torch.optim.SGD(momentum=0.9)`

> Duda
> Se podría usar linear interpolation?
>
> ```
>              __________________grad_actual_________________
> new_w = w + ( - (lr)(1-p)(derivative) + (p)(grad_anterior) )
> ```


##### Nesterov Momentum

### AdaGrad

(Adaptative lr) `torch.optim.Adagrad()`

### RMSProp

(Adaptative lr) `torch.optim.RMSprop()`

### Adam (Momentum + RMSProp)

(Momentun + RMSProp) `torch.optim.Adam()` The **best** and most used. 


## References
* https://sgugger.github.io/sgd-variants.html
* http://ruder.io/optimizing-gradient-descent/
* http://ruder.io/deep-learning-optimization-2017/
* http://www.fast.ai/2018/07/02/adam-weight-decay/
