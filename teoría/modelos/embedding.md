---
title: Representacion de palabras con vectores
autor: Javi
layout: post
dificultad: fácil
---

# Representacion de palabras con vectores

> #### Contenido
>
> * ¿Porqué representar las palabras como vectores?
> * ¿Como se consigue esta representación? Explicación del modelo teórico
> * Ahora te toca a tí. Crea tu modelo en Tensorflow

## ¿Como representar las palabras?

Vamos a ver una buena manera de cómo debemos representar las palabras para ponérselo fácil a una red neuronal. Tradicionalmete las palabras se han codificado como una lista ordenada de números. Estos índices no aportan información sobre el significado de la palabra o su relación con otras palabras. Esto es lo que vamos a solucionar usando vectores.

## ¿Porqué vectores?

Los vectores, son listas de numeros que

![vector](img/vector.png)

Cada palabras, tiene un significado (o varios) y una relación de similitud con el resto de palabras. Lo que vamos a hacer es asignarle varios números (un vector) a cada palabra de manera que cada número representa una dimensión. No te preocupes si esto de la dimensión no lo entiendes, lo que debes saber es que a cada palabra vamos asignarle unos numeros que nos aporten significado sobre esa palabra y su relación con otras palabras. Esto es lo que se conoce como **word embedding**.

## ¿Como se consigue esta representación?

Esta representación se consigue con, sorpresa, ¡una red neuronal!

Tradicionalemete se hacía contando cuantas veces aparece una palabra junto a sus palabras vecinas y luego hacer estadística. Pero podemos hacer una red neuronal que directamete haga el trabajo por nosotros. La idea es muy sencilla, veámosla.

Queremos una representación en forma de vector (una lista de 32 numeros mas o menos) por cada palabra.

![training](img/training.png)

## Modelo inverso

El modelo que dada una predice un conjunto de palabras similares es Skip-Gram.
 * Mejor en datasets grandes
 * Mejor para mapear palabras?

El modelo que dado un conjunto de palabras predice una palabra similares es Bag-of-Words model (CBOW).
 * Mejor en datasets pequeños
 * Suaviza más?
 * Util en la prediccuión de texto?
 
## Notas a tener en cuenta

Probablemente querramos coger solo las palabras más frecuentes (inglés: 1000.000.000) (español: 1500.000.000)
Un tamaño adecuado para el vector es 500

Las palabras poco comunes no se aprenden muy bien porque son poco frecuentes. Razón: Una red neuronal necesita ver varias veces el dato para aprendiendo bien, si unas clases se ven mucho más que otras, algo falla (datos no balanceados)


## Fuentes

* [Playing with word vectors](https://medium.com/swlh/playing-with-word-vectors-308ab2faa519)
* [tensorflow](https://www.tensorflow.org/tutorials/word2vec)
* [blog](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors)
* [otro blog](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)
