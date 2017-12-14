# Autoencoders

* Es una red neuronal donde la entrada y la salida **es la misma**.
* Las capas ocultas codifican la información.

Un autoencoder es una aproximación de la función de identidad, y si ponemos restricciones como reducir el número de neuronas en la capa oculta, podemos comprimir la información y así descubrir patrones interesantes en la información.

Se pueden hacer autoencoders con solo 1 capa oculta o con más.

![imagen](https://deeplearning4j.org/img/deep_autoencoder.png)

#### Utilidad

Para tener una representación de menos dimensiones de los datos.

A veces nuestros datos, tienen datos redundates y utilizar los datos así en una red neuronal significa más parametros a aprender, más computación, y más probabilidad de overfitting.

Si puedes recustruir bien el dato con el autencoder, significa que ese autoencoder funciona bien y es fiable.

Entoces podrás usar la codificación del dato porporcionada por el autoencoder, como entrada para una futura red neuroal.

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
