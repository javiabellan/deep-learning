# Autoencoders

* Es una red neuronal donde la entrada y la salida **es la misma**.
* Las capas ocultas codifican la información.

Un autoencoder es una aproximación de la función de identidad, y si ponemos restricciones como reducir el número de neuronas en la capa oculta, podemos comprimir la información y así descubrir patrones interesantes en la información.

Se pueden hacer autoencoders con solo 1 capa oculta o con más.

![imagen](https://deeplearning4j.org/img/deep_autoencoder.png)

## Utilidades

Comprimir datos. Tener una representación de menos dimensiones de los datos. Esto se puede aplicar a:

#### Formatear datasets

A veces nuestros datos, tienen datos redundates y utilizar los datos así en una red neuronal significa más parametros a aprender, más computación, y más probabilidad de overfitting. Si puedes recustruir bien el dato con el autencoder, significa que ese autoencoder funciona bien y es fiable. Entoces podrás usar la codificación del dato porporcionada por el autoencoder, como entrada para una futura red neuroal.

#### Buscar imágenes similares

Como vamos a ver, podemos comprimir imagens es un vector de 30 (MNIST). Este vector se podría usar para buscar imágenes parecidas.

#### Buscar canciones similares ???

#### Comprimir datos de forma semática

No solo se comprimen datos, sino que una vez comprimidos mantienen su significado en forma de vector: Más información buscar el paper de semantic hashing de Geoff Hinton.

#### Modelado de tema (Topic Modeling & Information Retrieval)

Pasar documentos de texto a "bag of words" (Probabilidad de ocurrencia de cada palabra) y usar para buscar documentos por tema concreto.


## Ejemplo

Vamos a hacer un autoncoder para el MNIST.

Las capas del codificador serán `784 (input) ----> 1000 ----> 500 ----> 250 ----> 100 -----> 30`

Usar 1000 en la primera capa oculta es algo que parece mal pero es bueno, NO SE PORQUE

In this case, expanding the parameters, and in a sense expanding the features of the input itself, will make the eventual decoding of the autoencoded data possible.

This is due to the representational capacity of sigmoid-belief units, a form of transformation used with each layer. Sigmoid belief units can’t represent as much as information and variance as real-valued data. The expanded first layer is a way of compensating for that.

El decodificador es igual pero a la inversa: `30 ----> 250 ----> 500 ----> 1000 ----> 784 (output)`

> NOTA
> At the stage of the decoder’s backpropagation, the learning rate should be lowered, or made slower: somewhere between 1e-3 and 1e-6, depending on whether you’re handling binary or continuous data, respectively.

## Denoising Autoencoders

Autoencoders with more hidden layers than inputs run the risk of learning the identity function (the output simply equals the input) thereby becoming useless.

Denoising autoencoders are an extension of the basic autoencoder, and represent a stochastic version of it. Denoising autoencoders attempt to address identity-function risk by randomly corrupting input (i.e. introducing noise) that the autoencoder must then reconstruct, or denoise.

## Restricted boltzmann machine

Las RBMs son un concepto similar a un autoencoder con 1 capa oculta pero con las siguientes diferencias:

* La parte de codificar y descodificar es la misma, es decir, es como "que va y vuelve"
* Usa un enfoeque estocástico (los coefifcites son iniciados de forma aleaoria)
* Son útiles para sistemas de recomendación
* Cundo tiene varias capas ocultas son las Deep-belief networks

![RBM](https://deeplearning4j.org/img/multiple_hidden_layers_RBM.png)

"A la vuleta" se multiplican por los mismos pesos que a la ida, pero los bias son nuevos.

![RBM2](https://deeplearning4j.org/img/reconstruction_RBM.png)

## Referencias

#### Autoencoder
* [Stanford: Autoencoder](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
* [deeplearning4j: Autoencoder](https://deeplearning4j.org/deepautoencoder)

#### Restricted boltzmann machine
* [deeplearning4j: RBM](https://deeplearning4j.org/restrictedboltzmannmachine)
* [Introduction to Restricted Boltzmann Machines](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/)
