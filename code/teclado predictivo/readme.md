---
title: Predicción de palabras
Autor: Javi
---

# Predicción de palabras

* Repaso de teoría
  * Redes neuronales recurrentes (LSTM)
* Implementación en Tensorflow
  * Obtener y limpiar datos
  * Construir diccionario
  * Definir y entrenar el modelo
  * Probarlo con nuestras frases
* Posibles mejoras
  * Word embeddings
  * Encoder-Decoder


## Introducción

La predicción de palabras se basa en adivinar la próxima palabra más proble en base a las palabras anteriores. Este predictor se puede incorporar a un teclado para poder predecir las palabras siguientes.

Los datos en que se basa para realizar dicha predicción son:

* Las palabras que has escrito anteriormete.
* Un conocimieto previo aprendido sobre tu forma de escribir (lenguaje, estilo, etc).

Se puede ir un poco más allá con estos truquillos:

* Se puede predecir más de una palabra, si la palabra predicha la tomamos como real y volvemos a predecir una nueva.
* Se puede predecir la palabra que tienes a medio medio escribir, si comparamos nuestro conjunto de predicciones con la palabra que estás escribiendo.


## Teoría

¿Que tipo de red neuronal se necesita para cuando tenemos que recordar una serie de entradas (en este caso palabras) para hacer una predicción sobre lo que hemos visto? La respuesta son las redes neuronales recurrentes porque son capaces de recordar. En concreto hay un tipo de red recurrente que funciona especialmente bien, las LSTM.

La entrada de una neurona RNN, es la entrada normal + la salida suya de antes:

`salidaActual = act( (entradaActual + salidaAnterior) * pesos)`

El problea de las RNN es que el problema de **vanishing and exploding gradient** les afecta mucho debido a su recursividad. Por eso usaremos untipo especial de RNN: las **LSTM**, que guradan un estado en su interior. Las LSTM superan el problema del vanishinh/exploding gradient al mantener el error que debe ser propagado hacia atrás en el agoritmo de backpropagation.


## Datos

Lo primero que tenemos que hacer es conseguir unos buenos datos, vamos a descargar un libro de Shakespeare.

Vamo a usar el libro *Más allá del bien y del mal* de Friedrich Nietzsche para entrenar nuestra red. Y pasamos todas las letras a minúscula para simplificar.

```python
ruta = 'nietzsche.txt'
texto = open(ruta).read().lower()
print('Número de letras:', len(text))
```

Vemos que tenemos un texto de 600893 letras para entrenar la red.

Una vez que obtenemos los datos vamos a limpiarlos un poco quitar signos de puntuación raros y todo eso.

```python
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()
```


## Construir el diccionario

La red neuronal entiende números, no palabras. Por lo tanto hay que convertir las palabras de alguna forma.

Aquí estamos utilizando la técnica del diccionario, donde a cada palabra le asignamos un índice. Lo hacemos así, porque es fácil y tenemos pocas palabras (se necesita una salida distinta para cada palabra)

Otra opción, probalemente mucho mejor, es usar word emmbedings. De esta forma las palabras se codifican a un vector de tamaño fijo que aporta información sobre el significado de la palabra.


Array de parejas que cuanta cuantas veces aparece cada palabra. Además están oredeanads descententemete (primero las más frecuentes).

```python
count = collections.Counter(words).most_common()
[ (',', 14), ('the', 11), ('.', 8), ('and', 7), ...
```

El dictionary, aprovechara el orden antarior, para asignar índices únicoas a cada palabra (el 0 para la palabra mas frecuente, y asi).

```python
dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

#     dictionary:         { ',':0,  'the':1,  '.':2,  'and':3, ...
#     reverse_dictionary: { 0:',',  1:'the',  2:'.',  3:'and', ... 
```
    
## Convertir vocabulario a vectores de palabras

```python
s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)
```

## Construir el modelo

```python
# TODO
```

## Entrenar el modelo

```python
# TODO
```

## Probar el modelo

```python
# TODO
```

## Otras arquitecturas a considerar:

* Método tradicional: Wikipedia: Markov chain, y, language model
* Método DL: https://arxiv.org/pdf/1709.06429.pdf
* CNN: 20 * conv layer with kernel size=5, dimensions=300

## Referencias
* Predecir palabra, link 1, (sencillo) [Medium](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537), [Github](https://github.com/roatienza/Deep-Learning-Experiments/tree/master/Experiments/Tensorflow/RNN)
* Predecir palabra, link 2, diferente[Github](https://github.com/dipendra009/Text-generation)
* [Predecir solo letra](https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218)
