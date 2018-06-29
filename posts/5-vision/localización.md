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

