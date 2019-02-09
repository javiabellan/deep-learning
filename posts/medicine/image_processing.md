> # TODO
>
> - [Procesamiento imgs Ginés](http://dis.um.es/~ginesgm/pi.html)
> - [DLTK](https://github.com/DLTK/DLTK)
> - [MedicalTorch](https://medicaltorch.readthedocs.io)
> - [Albumentations](https://github.com/albu/albumentations)
> - [Imgaug](https://github.com/aleju/imgaug)

- Operaciones globales
  - Mejora del contraste
  - transf. de color
  - histogramas
- Filtros y convoluciones
  - Suavizado
  - Perfilado
  - Detección de bordes
  - Características
- Transoformaciones geométricas
  - Escalado
  - Selección de ROI (regiones de interés)
  - Deformaciones
- Transformaciones lineales (ya no son posiciones en el espacio)
  - Transformadas Forier (FFT)
  - DCT (transoforada de coseno)
  - Wavelet

> CAD Compute Aid Detection/Diagnostic

### Operaciones globales

Procesar pixeles individualmente. New_mage(x,y):= f(Image(x,y))
- Invertir: 255 – Image(x,y)
- Suamr: Aclarar (hay saturación)
- Restar: Oscurecer (hay saturación)
- Muliplicar: Aumetar el contraste (hay saturación)
- Dividir: Disminuir contrarte (hay perdida de color)
- Gama: Oscurece y aclara sin saturar

Histograma: Vemos si:
- Es muy oscura/clara la imagen
- Esta muy o poco contrastada
Una buena imagen tiene un histograma bastante bien repatido

Curva tonal vemos la operación en trada salida.
