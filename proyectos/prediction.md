---
link: https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218
---

# Predicción de palabras

Vamos a predir la palabra en base a las palabras anteriores. Usaremos una red recurrente pq es capaz de memorizar cosas.

La entrada de una neurona RNN, es la entrada normal + la salida suya de antes:

`salidaActual = act( (entradaActual + salidaAnterior) * pesos)`

El problea de las RNN es que el problema de **vanishing and exploding gradient** les afecta mucho debido a su recursividad. Por eso usaremos untipo especial de RNN: las **LSTM**, que guradan un estado en su interior. Las LSTM superan el problema del vanishinh/exploding gradient al mantener el error que debe ser propagado hacia atrás en el agoritmo de backpropagation.

## Datos

Vamo a usar el libro *Más allá del bien y del mal* de Friedrich Nietzsche para entrenar nuestra red. Y pasamos todas las letras a minúscula para simplificar.

```python
ruta = 'nietzsche.txt'
texto = open(ruta).read().lower()
print('Número de letras:', len(text))
```

Vemos que tenemos un texto de 600893 letras para entrenar la red

## Modelo

Concretamente vamos a pasarle un cadena de **40** letras y la red tendrá que predecir la siginte letra,


## Otras arquitecturas a considerar:

* Método tradicional: Wikipedia: Markov chain, y, language model
* Método DL: https://arxiv.org/pdf/1709.06429.pdf
* CNN: 20 * conv layer with kernel size=5, dimensions=300
