# Traductor

## Teoría necesaría
* wordEmd
* RNN, LSTM
* [Sequence to sequence](/teoría/seq2seq.md)

## Introducción

En tiempos primitivos, la traducción de un texto, se hacía dividiendolo en trozos individales, traducirlo y luego juntarlo todo. Esto producía traducciones algo pobres e inconexas.

Pero con la llegada de las redes neuronales recurrentes, la red va leyendo el texto y se va haciendo un representación interna de lo que lee, para luego traducirlo. Esta forma es mucho más parecida a como lo hacemos los humanos.

## Funcionamiento

En concreto la parte que lee y va entendiendo, es el codificador, (en color azul). Éste representa lo que va entendiendo en forma de un vector de números. Finalemente cuando acaba de leer, la traducción resultante la genera el decodificardor (en color rojo) a partir del vector "de entendimiento".

![img](https://github.com/tensorflow/nmt/blob/master/nmt/g3doc/img/encdec.jpg)

Esta estructura de red neuronal se llama [**sequence to sequence**](/teoría/seq2seq.md) y es muy útil para traducir porque separa la parte de entender de la parte de traducir. Por ejemplo, imagina que tenemos un traductor de chino a español, si ahora queremos un traductor de chino a inglés, el codificador que entiende chino ya lo tenemos, ahora sólo nos falta el decodificador que traduce a inglés.

## Vayamos por partes

### ¿Como lee y entiende las palabras?

Fijate en cualquer texto, es solo una lista de palabras, y cada palabra es un lista de letras. Para tí la palabras tienen sentido, pero para un ordenador, sólo son letras detrás de otras. Por lo tanto parece que antes que aprender a traducir, habrá que aprender a leer y conocer el significado de las palabras de un idioma.

Dar significado a las palabras, es un proceso previo que se debe hacer y se conoce como **word embedding**. Consiste en crear una representación en forma de vector de cada palabra, de forma que esos números apoerten información sobre lo que esa palabra representa. Si quieres conocer más sobre word embedding ve aquí.

En nuestro caso vamos a coger un word embbeding ya entrenado por alguien.

### Ya entiendo las palabras, ¿ahora como las leo?

Al igual que los humanos, una [red neuronal recurrente](/teoría/redes/rnn.md), va leyendo palabra por palabra y se va haciendo una idea. Esta idea que se hace es la salida de la red que será el vector "de entendimiento".


## Referencia

https://github.com/tensorflow/nmt

## Industria

 * DeepL
 * Google translator