---
title: Capsule netwoks
autor: Javi
date: 13-12-2017
paper: https://arxiv.org/pdf/1710.09829.pdf
layout: post
---

# Capsule networks

## Resumen

* Es una idea de Geoffrey E. Hinton.
* El objetivo Ser invariante a la posición en el reconocimiento de imágenes.
* En lugar de añadir mas capas, pone "capas dentro de otras".

## Problemas de las CNNs

Aunque las redes convolucionales funcionan bastante bien en el reconocimiento de imágenes, tienen algunos puntos débiles.

Una CNNs funciona de la sigiente manera:

1) En las primeras capas busca patrones simples como bordes.
2) Con los patrones encontrados, buscar patrones más complejos como ojos.
3) Con los patrones complejos encontrados, busca patrones todavía más complejos como caras.

#### Problema 1: Distribución espacial 

El poblema es que aunque sigue una jerarquía correcta, se va perdiendo la distribución espacial de los patrones. Y puede pensar que ve una cara cuando ve ojos y boca. Es un problema de falso positivo.

![image](https://cdn-images-1.medium.com/max/800/1*pTu8CbnA_MzRbTh6Ia87hA.png)

#### Problema 2: Rotar e invertir la imagen

Este es otro gran problema debido a que las imagenes que se usan para entrenar, suelen tener simpre en la misma posición. Por lo tanto los filtros se aprenderán para ese enfoque. Asi que conforme rotamos la imagen, la CNN dejará de predecir lo que ve. Es un problema de falso negativo.

Una solución común es utilizar imagenes con distista rotación al entrenar la red, pero esto implica aprender los mismos patrones para diferentes posiciones, una solución algo ineficiente en cuanto el numero de patrones a aprender.

#### Problema 3: Hackeo de pixeles

Además exite un tercer problema (inherente a todas las redes neuronales pero que afecta especialmete a las CNNs) que consiste en modificar lijeramente la imagen para engañar a la red neuronal. Esta modificación consiste en buscar y modificar algunos pixeles de forma concienzuda para engañar a la red, pero visaulmente la imagen no cambia casi nada para un humano. Es un problema de falso negativo.

## Conclusión de las CNNs 

Debido a estos 3 problemas, Hinton dice que las CNNs están condenadas. Y propone las Capsule netwoks

> “Convolutional neural networks are doomed” — Geoffrey Hinton

## Más información

#### Oficial

* [Paper](https://arxiv.org/abs/1710.09829)
* [Paper (PDF)](https://arxiv.org/pdf/1710.09829.pdf)
* [Otro paper](https://openreview.net/pdf?id=HJWLfGWRb)

#### Medium

* [Understanding Hinton’s Capsule Networks. Part I: Intuition.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
* [Understanding Hinton’s Capsule Networks. Part II: How Capsules Work.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66)
* [Understanding Hinton’s Capsule Networks. Part III: Dynamic Routing Between Capsules.](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-iii-dynamic-routing-between-capsules-349f6d30418)
* [Capsule Networks Are Shaking up AI](https://hackernoon.com/capsule-networks-are-shaking-up-ai-heres-how-to-use-them-c233a0971952)
