# Residual network

If the CNN is to deep (+10 layers) it will hava problems to train.


```python
def ResnetLayer(x):
    return x + ConvBachnormLayer(x)  # 1 or 2 convolutions
```

![resnetBlock](https://cdn-images-1.medium.com/max/1600/1*pUyst_ciesOz_LUg0HocYg.png)


```
ResnetLayer =  x + ConvBachnormLayer(x)
```

is equals to

```
                        ___residual___ = error
ConvBachnormLayer(x) =  ResnetLayer - x
```

So each convolution shows how much error.

Boosting.


## References
* https://youtu.be/H3g26EVADgY
* [Understand Deep Residual Networks](https://blog.waya.ai/deep-residual-learning-9610bb62c355)
