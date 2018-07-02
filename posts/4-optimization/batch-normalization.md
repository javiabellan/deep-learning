# Batch Normalization

## Inital problem

Activations are getting bigger and bigger or smaller and smaller every layer. You have not control over this.

Sugerencia mia: (because you are using ReLU, not the sigmoid)

## Naive Solution

**Normalize activations**. But this literally does nothing 
because SGD will undo it in the next minibatch repeatedly.

```python
def naive_batchnorm(x):
    return (x-x.mean()) / x.std()
```

## Solution

**Normalize activations** and give 2 learneble parameters to modify the data.
* One parameter to scale the output (multiplication)
* Other parameter to shift the output (sum)

So now the network has this to parameters to modify the output at once,
and not scale and shift every single activation

```python
class bachnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size_output))
        self.shift = nn.Parameter(torch.zeros(size_output))
        
    def forward(self, x):
        return ((x-x.mean()) / x.std()) * self.scale + self.shift
```


> ### Bachnorm before or after ReLU?
> Original paper put bachnorm and then ReLU, but it works a bit better **ReLU and then bachnorm**.
>
> Because if you put bachnorm and then ReLU, you are truncating normalization at zero, loosing information.

## Why is also a regularizer?

Each minibatch is going to have a different mean() and a different std() to the previous minibath.
So they add noise, and when you add noise, in any kind, it regularizes your model




## References

* https://youtu.be/H3g26EVADgY
* https://towardsdatascience.com/understanding-batch-normalization-with-examples-in-numpy-and-tensorflow-with-interactive-code-7f59bb126642
* https://jermwatt.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_3_Batch_normalization.html
