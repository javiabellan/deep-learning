# Ver lo que una CNN ha aprendido: [Inceptionism](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

## Prerequisitos:

 * [Paper CNN](https://arxiv.org/pdf/1603.07285.pdf)
 * [Deconvolutions](https://arxiv.org/ftp/arxiv/papers/1609/1609.07009.pdf)
 * deep belief network (DBN)
 * [blog extra](https://rajpurkar.github.io/mlx/visualizing-cnns)

## INVERTIR RED: mostrar lo prendido para una clase dada

 * class logits (pre softmax) (produce mejores imagenes)
 * class probabilities (post softmax)

So here’s one surprise: neural networks that were trained to discriminate between different kinds of images have quite a bit of the information needed to.

Ver platano-> reconocer que un platano
Piensa en un platano -> Generar imagen mental de un platano

Imagen -> concepto
concepto -> visualizar imagen


## MAXIMIZAR LAS CARACTERISTICAS QUE VEA

 * Layer/DeepDream

Coger una capa entera y usarla para maximizar caraxterísticas.
De imagen (o ruido), pintar encima lo que cree que ve.

Segun la capa:

 * Bajas: Pintan patrones simples (raras trazos)
 * Altas: Pintan patrones complejos (animales, paisajes) (Inceptionism, deep dream)

#### Extra: Iteraciones:

If we apply the algorithm iteratively on its own outputs and apply some zooming after each iteration, we get an endless stream of new impressions, exploring the set of things the network knows about. We can even start this process from a random-noise image, so that the result becomes purely the result of the neural network
