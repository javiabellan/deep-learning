# Glosario

## Nivel 1
* Red neuronal artificial
* Red neuronal convolucional
* Backpropagation
* Bias
* Gradient descent
* Epoch: Cuando se pasa todo el training set (foward y backward) una vez
* Neuron
* Perceptrón
* Weight

## Nivel 2
* Batch
* Stochastic gradient descent

### Batch
Como pasar todos los datos de golpe en una epoch, es difícil (no cabe en memoria), se lo pasamos por grupos o **batches**.

A ver, queremos pasarle todos los datos y varias veces (varias epochs) a la red para que aprenda.
Imagina que tenoemos 2000 datos, pero solo caben 500.
pues habrá que hacer 4 **iteratios** con un **batch size** de 500, para completar 1 **epoch**

### Gradient descent
a ver, hay 3 tipos de gradient descent:

https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/

### Stochastic gradient descent (SGD)
Usar mini-batches. En lugar de actualizar los parametros de la red por cada muestra, hacerlo por grupos


## Nivel 3
* Adagrad: Una variente de SGD
* Batch normalization
* Momentum: Una variente de SGD



### Batch normalization

Problema:
La distribución de los inputs de cada capa cambia durante el entrenamiento,
conforme los parámetros de la capa anterior cambian.

Esto relentiza el entrenamiento porque se requiere learning rates pequeños
y una cuidadosa inicialización de los paramatros.

Este problema se conoce como *internal covariate shif*.

[Paper](https://arxiv.org/abs/1502.03167)
