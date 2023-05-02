# Red neuronal recurrente

Las redes recurrentes o RNN se usan cuando la naturaleza es de los datos es sequencial, y por lo tanto se necesita recordar lo que se ha visto previamente. Este tipo de datos son el **texto**, el **audio** y el **vídeo**. Las redes neuronales normales no pueden procesar esta información (a no ser que les pasemos todos los datos de golpe).

<p align="center">
<img width="20%" src="http://www.realjabber.org/r3tf/totalconversation.png" />
</p>

Una neurona recurrente se basa en que parte de su salida se retroalimenta de nuevo como una entrada nueva. Pero la forma habitual de dibujarla es de forma "desenrollada", donde podemos ver la misma neurona en los distintos instantes de tiempo como se va actualizando con la informacion nueva de su salida anterior.

<p align="center">
<img width="60%" src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" />
</p>

En la vista desenrollada vemos como se van introduciendo los datos poco a poco (x0, x1, x2). Por ejemplo, para precesar texto, cada dato es una palabra. O para procesar video, cada dato es una imagen. Y la entrada que recive de su salida anterior, es una especie de memoria para recordar lo que ha visto anteriormente.

Este es la idea básica pero existen distintos tipos de RNN varían en términos de:

* Tipo de neurona: RNN básica, LSTM, GRU
* Direccionalidad: Unidireccional o bidireccional
* Profundidad: Número de capas


## Generative RNNs (Seq2Seq)

Esta formado por dos RNN, un codificador y un decodificador. El codificador se encarga de "entender" la información de entrada y plasmarlo en un vector. Y el decodificador usa ese vector de información para generar una nueva sequencia de datos (normalmente texto).

![img](http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png)

La ventaja de estar separado en dos partes (codificador y decodificador) es que ambas partes son independientes. Por ejemplo, imagina el problema de la traducción, tenemos un red seq2seq que traduce del chino al ingles, el codificador se encarga de entender el chino, y el decodificador se encarga de traducir lo que se ha dicho al inglés. Si ahora queremos un traductor de chino a español solo tendremos que cambiar el decodificador, porque el codificador de entender chino ya la tenemos.


#### Aplicaciones
- Traducción (de un idoma a otro)
- Diálogo (de preguntas a respuestas)
- Resumen de texto (de texto largo a texto corto)
- Transcripción de voz (de voz a texto)
- Y cualquier otro problema que se trate de pasar algo a otra cosa!

#### Referencias
- https://blog.suriya.app/2016-12-31-practical-seq2seq/
- https://guillaumegenthial.github.io/sequence-to-sequence.html
- https://towardsdatascience.com/sequence-to-sequence-tutorial-4fde3ee798d8
- https://www.tensorflow.org/tutorials/seq2seq
- https://github.com/tensorflow/nmt
- https://arxiv.org/abs/1703.01619
- https://arxiv.org/abs/1409.3215
- https://arxiv.org/abs/1406.1078




## Direccionalidad

- RNN unidireccional: Util para codificar pero tambien para generar. Similar a un transformer decoder autoregresivo (GPT).
- RNN bidireccional: Solo sirve para codificar leyendo de der a izq y biceversa. Similar a un transformer encoder (BERT).

## Entrenamiento

La forma de entrenar una red recurrente es con Backpropagation pero con un pequeño ajuste llamado **Backpropagation Through Time (BPTT)**.

## Referencias

- Andrew karpathy: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness)
- Chris Olah: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs)
- [wildml](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns)
