# Batch Normalization

## Inital problem

Activations are getting bigger and bigger or smaller and smaller every layer. You have not control over this.

Sugerencia mia: (because you are using ReLU, not the sigmoid)

## Naive Solution

**Normalize activations for every layer**. But this literally does nothing 
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
        self.m = nn.Parameter(torch.ones(size_output))
        self.a = nn.Parameter(torch.zeros(size_output))
        
    def forward(self, x):
        return ((x-x.mean()) / x.std()) * self.m + self.a
```

## References

* https://youtu.be/H3g26EVADgY
* https://towardsdatascience.com/understanding-batch-normalization-with-examples-in-numpy-and-tensorflow-with-interactive-code-7f59bb126642
* https://jermwatt.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_3_Batch_normalization.html
