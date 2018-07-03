# SGD Optimization


To train our neural net we detailed the algorithm of Stochastic [Gradient descent](/posts/1-basics/gradient_descent.md).
To make it easier for our model to learn, there are a few ways we can improve it.

### Gradient Descent

Just to remember what we are talking about,
the basic algorithm consists in changing each of our parameter where:
* `w` represents one of our **parameters** (weights or bias)
* `w.grad` is the **gradient** of the loss with respect to `w`
* `lr` is an hyperparameter called **learning rate**

```python
new_w = w - lr * w.grad
```

### Gradient Descent with Momentum

To accelerate thigs up, one improvement is to use mumentum that sum to the actual step, a little bit of the previous step.
*  `p` is a new hyperparameter called **momentum** often equals to 0.9 (other common values are .5, .7, .9 and .99).
```
             _______________grad_actual_______________
new_w = w + ( - (lr)(derivative) + (p)(grad_anterior) )
```

```python
new_v = lr * w.grad + p * v
new_w = w - new_v
```


In Pytorch: `torch.optim.SGD(momentum=0.9)`

> Duda
> Se podría usar linear interpolation?
>
> ```
>              __________________grad_actual_________________
> new_w = w + ( - (lr)(1-p)(derivative) + (p)(grad_anterior) )
> ```


### Nesterov Momentum

```python
for p in parameters:
    p1 = p - beta v[p]
model.reevaluate_grads()
for p in parameters
    v[p] = beta * v[p] + lr * p.grad
    p = p - v[p]
```

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
