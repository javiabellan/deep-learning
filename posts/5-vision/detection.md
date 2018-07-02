# Object detection

## Class Activation Maps (CAM) [fast.ai lesson 7](https://youtu.be/H3g26EVADgY?t=2h9m)

See the last conv layer ativations

¿No tendría la ultima capa de la CNN, un mapa de cuales son las caracterísitcas que ve?


## Single-object detection [fast.ai lesson 8](https://youtu.be/Z0ssNAbe81M)


## Multi-object detection [fast.ai lesson 9](https://youtu.be/0frKXR-2PBY)


### Bad solution: Brute force and R-CNN

1. Coge un clasificador CNN ya entrenado
2. Pasa ese clasificador a lo largo de distitas regiones de tu imagen

### Good solution: Look only once at your image

The final layer of the resnet is 512 channles of 7x7 (7x7x512)

If we want to predict up to 16 bounding boxes with its categories we have to options:

 * Put one `nn.linear()` layer with output size of 16*(4+clases). (YOLO)
 * Put one `nn.conv2d(stride=2)` layer with output size of 4x4x256. (SSD)

[YOLO](https://pjreddie.com/darknet/yolo/)

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

