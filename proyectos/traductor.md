# Traductor

## Teoría necesaría
wordEmd, RNN, LSTM, Seq2seq;

## Introducción

En tiempos primitivos, la traducción de un texto, se hacía dividiendolo en trozos individales, traducirlo y luego juntarlo todo. Esto producía traducciones algo pobres e inconexas.

Pero con la llegada de las redes neuronales recurrentes, la red va leyendo el texto y se va haciendo un representación interna de lo que lee, para luego traducirlo. Esta forma es mucho más parecida a como lo hacemos los humanos.

## Funcionamiento

En concreto la parte que lee y va entendiendo, es el codificador, (en color azul). El codificador representa la información que va leyendo  en forma de un vector de números. Finalemente cuando ya se ha leído el texto, la traducción resultante la genera el decodificardor (en color rojo) a partir del vector.

![img](https://github.com/tensorflow/nmt/blob/master/nmt/g3doc/img/encdec.jpg)

Esta estructura de red neuronal se llama **sequence to sequence** y es muy útil para traducir porque separa la parte de entender de la parte de traducir. Por ejemplo, imagina que tenemos un traductor de chino a español, si ahora queremos un traductor de chino a inglés, el codificador que entiende chino ya lo tenemos, ahora sólo nos falta el decodificador que traduce a inglés.


## Referencia

https://github.com/tensorflow/nmt

## Industria

 * DeepL
 * Google translator
