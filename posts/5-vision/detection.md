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

If we want to predict up to 16 bounding boxes with its categories we have to options for the **final layer**:

 * A `nn.linear()` layer with a 1D output of size of 16*(4+clases). ([YOLO](https://pjreddie.com/darknet/yolo/))
 * A `nn.conv2d(stride=2)` layer with a 3D output of size of 4x4x(4+clases). (SSD) (a bit better)


### Anchor boxes

```python
anc_grids  = [4,2,1]                       # Num of grid cells: 4x4 + 2x2 + 1 = 21

anc_zooms  = [0.7, 1., 1.3]                # different sizes:         3
anc_ratios = [(1.,1.), (1.,0.5), (0.5,1.)] # different aspect ratios: 3
```

21 * 3 * 3 = **187** final bounding boxes to select the best ones (>40% creo)

TODO Definir **receptive field**: Lo que ve cada anchor box de la imagen original


### Loss function

Process (for every image):

0. Get **prediction** and **target**:
   * Ground truth: 2D tensor `num_objs*(4+classes)`
   * Prediction: 2D tensor `16*(4+classes)`
1. **Matching part**: Determine which ground truth box is closest to wich anchor box (grid cell) (remeber we have several bboxes per grid cell)
2. **Compute loss**: Then is trivial, is the same as single object setection `detection_loss()` (L1 for bbox regression and CE for classification)

### History (papers)
1. (2013) Scalable object detection using deep neural networks (Multibox)
   * Loss function with mathing process
2. Faster R-CNN: Towards real-time object detection with region proposal networks
   * Add a CNN for region proposal (new idea)
   * Use that regions for clasification with a standar CNN
3. You Only Look Once: unified, real-time object detection
   * Same peroformace as Faster R-CNN but in one step
4. SSD single shot multibox detector
   * Same peroformace as Faster R-CNN but in one step
5. [Focal loss for dense object detection](https://arxiv.org/pdf/1708.02002.pdf) (RetinaNet)
   * Avoid the multiple messy bbox predictions

---

YOLO

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

