# Convolution Neural Network (CNN)


![img](https://cdn-images-1.medium.com/max/800/1*aBdPBGAeta-_AM4aEyqeTQ.jpeg)

- Inception v1
  - Different kernel sizes at the same level, to capture information of different sizes
  - With prior 1x1 convolutions to reduce number of channels (Bottleneck layer)
- Inception v2

- State of the art
  - Convolutions:
    - **Before**: Stride 1 convolution with pooling
    - **Now**: Stride 2 convolution without pooling (much faster)
  - Final layer:
    - **Before**: Fully connected
    - **Now**: Adaptative max pooling


### Bottleneck layer
 Let’s say you have 256 features coming in, and 256 coming out, and a 3x3 convolution.
 That is 256x256x3x3 convolutions that have to be performed (589,000s multiply-accumulate, or MAC operations).

Instead of doing this, we decide to reduce the number of features that will have to be convolved, say to 64 or 256/4:
- 256×64 × 1×1 = 16,000s
- 64×64 × 3×3 = 36,000s
- 64×256 × 1×1 = 16,000s

For a total of about 70,000 versus the almost 600,000 we had before. Almost 10x less operations!





### Residual Network (ResNet)

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



### Pytorch

```python
class ConvLayer(nn.Module):
    def __init__(self, input_chnls, output_chnls):
        super().__init__()
        self.conv = nn.Conv2d(input_chnls, output_chnls, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
    	return F.relu(self.conv(x))

class ConvNet(nn.Module):
    def __init__(self, layer_chns, classes):
        super().__init__()
        self.conv_layers = nn.ModuleList([ConvLayer(layer_chns[i], layer_chns[i+1])
            for i in range(len(layer_chns)-1)])
        self.fc = nn.Linear(layer_chns[-1], classes)
        
    def forward(self, x):
        for l in self.conv_layers: x = l(x) # Conv layers
        x = F.adaptive_max_pool2d(x, 1)     # Final pooling to 1 item
        x = x.view(x.size(0), -1)           # Flatten
        x = self.fc(x)                      # Fully connected
        return F.log_softmax(x, dim=-1)     # Softmax

model = ConvNet2([3, 20, 40, 80], 10)
```
## Disadvantages

1. Localización espacial. Como vamos mirando características de regiones pequeñas (primeras capas) a regiones más grandes (últimas capas) estamos perdiendo la relación espacial de objetos pequeños. Ejemplo:

	- No solo queremos saber que hay una nariz y una boca, queremos saber que la nariza se encuetra encima de la boca. No estoy seguro de esto porque en la capa siguiete habría directamente un filtro que reconozca caras.

2. Invariante a la rotación. No son inavariantes a la posición. Si rotamos un poco la imagen, o la invertimos, empieza a fallar.

	- Posible solución: Capsule network

## References

- [History of CNNs](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba)
- [Versions of inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
- [Resnet video](https://youtu.be/H3g26EVADgY)
- [Understand Deep Residual Networks](https://blog.waya.ai/deep-residual-learning-9610bb62c355)
