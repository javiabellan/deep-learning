---
autor: Javi
tipo: proyecto
teoría necesaria: [wordEmd, RNN, seq2seq]
---

# Traductor

## Introducción

En tiempos primitivos, la traducción de un texto, se hacía dividiendolo en trozos individales, traducirlo y luego juntarlo todo. Esto producía traducciones algo pobres e inconexas.

Pero con la llegada de las redes neuronales recurrentes, la red va leyendo el texto y se va haciendo un representación interna de lo que lee, para luego traducirlo. Esta forma es mucho más parecida a como lo hacemos los humanos.

## Resumen

En concreto la parte que lee y va entendiendo, es el codificador, (en color azul). Éste representa lo que va entendiendo en forma de un vector de números. Finalemente cuando acaba de leer, la traducción resultante la genera el decodificardor (en color rojo) a partir del vector "de entendimiento".

![img](https://github.com/tensorflow/nmt/blob/master/nmt/g3doc/img/encdec.jpg)

Esta estructura de red neuronal se llama [**sequence to sequence**](/teoría/modelos/seq2seq.md) y es muy útil para traducir porque separa la parte de entender de la parte de traducir. Por ejemplo, imagina que tenemos un traductor de chino a español, si ahora queremos un traductor de chino a inglés, el codificador que entiende chino ya lo tenemos, ahora sólo nos falta el decodificador que traduce a inglés.

## Repaso: Red neuronal recurrente (para acordarme de lo que leo)

Al igual que los humanos, una [**red neuronal recurrente**](/teoría/modelos/rnn.md), va leyendo palabra por palabra y se va haciendo una idea. Esta idea que se hace es la salida de la red que será el vector "de entendimiento".

In this tutorial, we consider as examples a deep multi-layer RNN which is unidirectional and uses LSTM as a recurrent unit. We show an example of such a model in Figure 2. In this example, we build a model to translate a source sentence "I am a student" into a target sentence "Je suis étudiant". At a high level, the NMT model consists of two recurrent neural networks: the encoder RNN simply consumes the input source words without making any prediction; the decoder, on the other hand, processes the target sentence while predicting the next words.

## Repaso: Word embedding (para entender la palabras)

Fijate en cualquer texto, es solo una lista de palabras, y cada palabra es un lista de letras. Para tí la palabras tienen sentido, pero para un ordenador, sólo son letras detrás de otras. Por lo tanto parece que antes que aprender a traducir, habrá que aprender el significado de las palabras de un idioma.

Dar significado a las palabras, es un proceso previo que se debe hacer y se conoce como [**word embedding**](/teoría/modelos/embedding.md). Consiste en crear una representación en forma de vector las palabras más comunes, de forma que esos números apoerten información sobre lo que esa palabra representa. (Las palabras más raras tendrán un vector comun indicando que la palabra es desconocida).

En nuestro caso vamos a coger los pesos de un word embeding ya entrenado como son word2vec o Glove. Pero en teoría, si tenemos una gran cantidad de datos, podemos hacer el word embedding nosotros mismos.

Los pesos que transorman las palabras a vectores será la primera capa de la red neuronal.

## El modelo

El modelo es muy simple, las capas son las siguientes:

1. Capa word embeding
2. Capa oculta 1: recurrente LSTM
3. Capa oculta 2: recurrente LSTM
4. Capa de proyección (word embeding inversa)

<p align="center">
<img width="50%" src="https://github.com/tensorflow/nmt/blob/master/nmt/g3doc/img/seq2seq.jpg" />
</p>

> #### Como interpretar la imagen
> * La red neuronal va de abajo (entrada) hacia arriba (salida).
> * Por un lado tenemos el codificador en azul a la izquierda y el decodificador en rojo a la derecha.
> * Cada fila horizontal de celdas, son la misma celda, pero en diferentes instantes de tiempo.
> * Los instantes de tiempo avanzan de izquierda a derecha.
> * Las flechas laterales indican la actualización del estado interno para una celda recurrente LSTM.
> * En la parte inferior (primera capa) se produce el word embedding.
> * En la parte superior (última capa) se produce el word embedding inverso.
> * Las palabra de salida, será la entrada en el siguiente instante de tiempo.
> * Las palabra `<s>`, indica que el texo ha finalizado y ya se puede empezar la traducción.



Según estemos entrenando o provando porporcinamos el texto traducido o no, pero siempre hay que poner una marca o separador al final del texto a traducir. Ejemplo:

* Una frase de entrenamiento: `I am a student _ Yo soy un estudiante`
* Una frase nueva para traducir: `You are a student _`


## Attention

Como tenemos un vector de representación interna con tamaño fijo, esto limita la cantidad de informción que podemos representar.
Esto significa que si tenemos un texto muy largo, no seremos capaces de representar toda su información.

Por lo tanto en lugar de codificar la secuencia en un solo vector. **Se codifica la secuencia en una sequencia de vectores y se van selecionando subconjuntos de vectores mientras se traduce.**

## Referencia

* https://github.com/tensorflow/nmt
* https://buzzrobot.com/machine-translation-without-the-data-21846fecc4c0
* [Paper sep 2014: Seq2seq](https://arxiv.org/abs/1409.3215)
* [Paper sep 2014: Seq2seq en traductor](https://arxiv.org/abs/1409.1259):
Propone seq2seq para traducir. El vector de representación es de tamaño fijo (esto limita la cantidad de información a represntar). Funciona bien para frases cortas, pero cuando el texto aumenta, la traducción empeora.
* [Paper sep 2014: Attention](https://arxiv.org/abs/1409.0473): Propone un vector de tamaño variable
* [Paper ago 2015: Mejora la Attention](https://arxiv.org/abs/1508.04025)
* https://www.iotforall.com/transformer-vs-deepL-attention-based-machine-translation/

## Industria

 * DeepL
 * Google translator
