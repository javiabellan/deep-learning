# Red neurnal recurrente

Las redes recurrentes o RNN se usan cuando la naturaleza es de los datos es sequencial, y por lo tanto se necesita recordar lo que se ha visto previamente. Este tipo de datos son el **texto**, el **audio** y el **vídeo**. Las redes neuronales normales no pueden procesar esta información (a no ser que les pasemos todos los datos de golpe).

<p align="center">
<img width="20%" src="http://www.realjabber.org/r3tf/totalconversation.png" />
</p>

Una neurona recurrente se basa en que parte de su salida se retroalimenta de nuevo como una entrada nueva. Pero la forma habitual de dibujarla es de forma "desenrollada", donde podemos ver la misma neurona en los distintos instantes de tiempo como se va actualizando con la informacion nueva de su salida anterior.

<p align="center">
<img width="60%" src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" />
</p>

En la vista desenrollada vemos como se van introduciendo los datos poco a poco (x0, x1, x2). Por ejemplo para precesar texto, cada dato es una palabra. O para procesar video, cada dato es una imagen. Y la entrada que recive de su salida anterior, es una especie de memoria para recordar lo que ha visto anteriormente.

Este es la idea básica pero existen distintos tipos de RNN varían en términos de:

* Tipo de neurona: RNN básica, LSTM, GRU
* Direccionalidad: Unidireccional o bidireccional
* Profundidad: Número de capas


## LSTM


## Direccionalidad

## Entrenamiento

La forma de entrenar una red recurrente es con Backpropagation pero con un pequeño ajuste llamado Backpropagation Through Time (BPTT).

## Referencias

* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): Explicación de Chris Olah sobre RNN y LSTM bastante buena
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [wildml](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
