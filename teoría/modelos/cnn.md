---
title: Redes neuronales convolucionales
autor: Javi
layout: post
dificultad: media
---

# Reconocimiento de imágenes: CNN


## Trucos computacionales

 * ReLU
 * dropout
 * batch normalization

## Problemas

1. Localización espacial. Como vamos mirando características de regiones pequeñas (primeras capas) a regiones más grandes (últimas capas) estamos perdiendo la relación espacial de objetos pequeños. Ejemplo:

	- No solo queremos saber que hay una nariz y una boca, queremos saber que la nariza se encuetra encima de la boca. No estoy seguro de esto porque en la capa siguiete habría directamente un filtro que reconozca caras.

2. Invariante a la rotación. No son inavariantes a la posición. Si rotamos un poco la imagen, o la invertimos, empieza a fallar.

	- Posible solución: Capsule network

