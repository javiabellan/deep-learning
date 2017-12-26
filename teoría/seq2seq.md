# Sequence to sequence


## Aplicaciones
* Traducción (de un idoma a otro)
* Diálogo (de preguntas a respuestas)
* Transcripción de voz (de voz a texto)
* Y cualquier otro problema que se trate de pasar algo a otra cosa!

## Idea principal

Esta formado por dos RNN, un codificador y un decodificador. El codificador se encarga de "entender" la información de entrada y plasmarlo en un vector. Y el decodificador "traduce" el vector de una forma determinada para que sea entendible.

![img](http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png)

La ventaja de estar separado en dos partes (codificador y decodificador) es que ambas partes son independientes. Por ejemplo, imagina el problema de la traducción, tenemos un red seq2seq que traduce del chino al ingles, el codificador se encarga de entender el chino, y el decodificador se encarga de traducir lo que se ha dicho al inglés. Si ahora queremos un traductor de chino a español solo tendremos que cambiar el decodificador, porque el codificador de entender chino ya la tenemos.

## Referencias

* https://towardsdatascience.com/sequence-to-sequence-tutorial-4fde3ee798d8
* https://arxiv.org/abs/1703.01619
