# Sequence to sequence

## Idea principal

Esta formado por dos RNN, un codificador y un decodificador. El codificador se encarga de "entender" la información de entrada y plasmarlo en un vector. Y el decodificador usa ese vector de información para generar una nueva sequencia de datos (normalmente texto).

![img](http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png)

La ventaja de estar separado en dos partes (codificador y decodificador) es que ambas partes son independientes. Por ejemplo, imagina el problema de la traducción, tenemos un red seq2seq que traduce del chino al ingles, el codificador se encarga de entender el chino, y el decodificador se encarga de traducir lo que se ha dicho al inglés. Si ahora queremos un traductor de chino a español solo tendremos que cambiar el decodificador, porque el codificador de entender chino ya la tenemos.


## Aplicaciones
* Traducción (de un idoma a otro)
* Diálogo (de preguntas a respuestas)
* Resumen de texto (de texto largo a texto corto)
* Transcripción de voz (de voz a texto)
* Y cualquier otro problema que se trate de pasar algo a otra cosa!

## Referencias

* https://guillaumegenthial.github.io/sequence-to-sequence.html
* https://towardsdatascience.com/sequence-to-sequence-tutorial-4fde3ee798d8
* https://www.tensorflow.org/tutorials/seq2seq
* https://github.com/tensorflow/nmt
* https://arxiv.org/abs/1703.01619

#### Papers oficiales

* https://arxiv.org/abs/1409.3215
* https://arxiv.org/abs/1406.1078
