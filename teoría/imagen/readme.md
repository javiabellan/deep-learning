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





## Capsule network (idea de Geoffrey E. Hinton)

Objetivo: Ser invariante a la posición.

En lugar de añadir mas capas, pone "capas dentro de otras"

![capsule](img/capsule.png)

[paper](https://arxiv.org/abs/1710.09829)
https://github.com/naturomics/CapsNet-Tensorflow


---


# Reconocimiento y detección de imágenes

No solo queremos saber lo que vemos sino que también dónde lo vemos.

## ¿Como se hace?

### Fuerza bruta y R-CNN

1. Coge un clasificador CNN ya entrenado
2. Pasa ese clasificador a lo largo de distitas regiones de tu imagen

### Solución mejor: [YOLO](https://pjreddie.com/darknet/yolo/)

You Only Look Once (a diferencia de la fuerza bruta y R-CNN)

1. Empieza dividiendo la imajen con una rejilla (15x15 por ejemplo)
2. Cada cuadradito puede prodicir 5 bouning boxes
3. Se generan muchas bounding boxes con una probabilidad determinda
4. Nos quedamos con las bounding boxes más seguras (poner un limite ej. >30%)

![yolo](img/yolo.png)

 * La estructura de YOLO es simplemente una CNN
 * Está escrito en darknet (C)
 * Ahora está YOLOv2 que va mejor
 * [YOLO paper](https://pjreddie.com/media/files/papers/yolo_1.pdf)
 * [YOLOv2 paper](https://pjreddie.com/media/files/papers/YOLO9000.pdf)


### Pensamiento proio

¿No tendría la ultima capa de la CNN, un mapa de cuales son las caracterísitcas que ve?










