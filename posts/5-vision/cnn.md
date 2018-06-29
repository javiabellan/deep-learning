# Convolution Neural Network (CNN)


### State of the art

Convolutions:
* **Before**: Stride 1 convolution with pooling
* **Now**: Stride 2 convolution without pooling (much faster)

Final layer:
* **Before**: Fully connected
* **Now**: Adaptative max pooling

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

