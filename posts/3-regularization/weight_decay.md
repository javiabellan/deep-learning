# Weight decay

Is a regularization method to reduce over-fitting
and consists in adding to the loss function some term in order to penalice high weights.

There are to types of weight dacay: L1 regularization and L2 regularization.


## L1 regularization

L1 regularization consists in adding to the loss function **the sum of the absolute value of all the weights** of the model,
multiplied by a `wd` hyper-parameter (usually 0.0005):

```python
l1_loss = loss + wd * all_weights.abs().sum() / 2   # / 2 is for simplify derivative
```

## L2 regularization

L2 regularization consists in adding to the loss function **the sum of the squares of all the weights** of the model,
multiplied by a `wd` hyper-parameter (usually 0.0005):

```python
l2_loss = loss + wd * all_weights.pow(2).sum()
```

If you remember SGD, we dont care about the loss function, **we care about its derivative**.

So because loss is now a sum, we ca take the derivative of each part separetly

```python
derivative(l2_loss) = derivative(loss) + derivative( wd * w^2)
=
derivative(l2_loss) = derivative(loss) + (2 * wd * w)
```

> #### Simplify the derivative
> Add `/2` to cancel out the `2` that appears in the dericative
>
> ```python
> l2_loss = loss + wd * all_weights.pow(2).sum() / 2   # / 2 is for simplify derivative
>
>derivative(l2_loss) = derivative(loss) + derivative( wd * w^2 / 2)
>=
>derivative(l2_loss) = derivative(loss) + (wd * w)
> ```
>
> So in a SGD update will be `w = w - lr * w.grad - lr * wd * w`
>
> In this equation we see how we subtract a little portion of the weight at each step, hence the name decay.

In practice, libraries nearly always implemented by adding `wd*w` to the gradients,
rather than actually changing the loss function:
we don’t want to add more computations by modifying the loss when there is an easier way.

So why make a distinction between those two concepts if they are the same thing?
The answer is that they are only the same thing for vanilla SGD,
but as soon as we add momentum, or use a more sophisticated optimizer like Adam,
L2 regularization (first equation) and weight decay (second equation) become different

## L2 regularization and weight decay in SGD with momentum

Using L2 regularization consists in adding wd*w to the gradients (as we saw earlier)
but the gradients aren’t subtracted from the weights directly. First we compute a moving average:

```python
# L2 regularization in SGD with momentum
moving_avg = alpha * moving_avg + (1-alpha) * (w.grad + wd*w)
```

…and it’s this moving average that will be multiplied by the learning rate and subtracted from w. So the part linked to the regularization that will be taken from w is lr* (1-alpha)*wd * w plus a combination of the previous weights that were already in moving_avg.

On the other hand, weight decay’s update will look like

```python
# Weight decay in SGD with momentum
moving_avg = alpha * moving_avg + (1-alpha) * w.grad 
w = w - lr * moving_avg - lr * wd * w
```
We can see that the part subtracted from w linked to regularization isn’t the same in the two methods.


## L2 regularization and weight decay in Adam

When using the Adam optimizer, it gets even more different:
in the case of L2 regularization we add this wd*w to the gradients then compute a moving average of the gradients and their squares before using both of them for the update. Whereas the weight decay method simply consists in doing the update, then subtract to each weight.






## References
* http://www.fast.ai/2018/07/02/adam-weight-decay/
