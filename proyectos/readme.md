# Proyectos

Lista de proyectos que puedes hacer. Están desarrollados en Python usando Tensorflow, consulta la [guía de instalación](/proyectos/instalación.md) para poner a punto tu máquina!

Los proyectos se dividen en categorías segun la naturaleza de los datos: texto, imagen, audio, etc. Además cada proyecto indica que tecnología usa, por si quieres echarle un vistazo antes de ponerte a programar.

## Texto

|      Proyecto        |      Datos       | Tecnología | Referencia |
|----------------------|:----------------:|:----------:|:----------:|
| [Teclado predictivo](/proyectos/teclado-predictivo) | texto -> palabra | word embedding, LSTM |  |
| [Traductor](/proyectos/traductor)    | texto -> texto traducido   | seq2seq    | [link](https://github.com/tensorflow/nmt) |
| Genera resúmenes    | texto -> texto resumido    | seq2seq    | |
| bot  | pregrunta -> respuesta     | seq2seq    | [link](https://github.com/vinhkhuc/MemN2N-babi-python)|
| Conversación (chatbot)   | pregruntas -> respuestas     | seq2seq    | [link](https://github.com/vinhkhuc/MemN2N-babi-python)|
| Reconocimiento de voz  | audio -> texto  | seq2seq    | |
| Responder preguntas sobre una imagen  | imagen -> texto  |     | [link](https://github.com/abhshkdz/neural-vqa) |
| Búsqueda de documentos (Semantic Search) | | | |
| Analizador de sentimientos (Twitter) | | | |
| Escritor de Wikipedia | | | |

#### Ideas
* [NLP Medium](https://codeburst.io/nlp-fundamental-where-humans-team-up-with-machines-to-help-it-speak-ac8c6dbaba88)
* (https://algorithmia.com/tags/text-analysis)

## Imágen

#### Reconocimiento (dime lo que ves)
* Reconocimiento de números (del 0 al 9)
* Reconocimiento de objetos
* [Descripción de imágenes](https://github.com/karpathy/neuraltalk2)
* [Cuenta una historia de la imágen](https://github.com/ryankiros/neural-storyteller)

#### Localización (dime lo que ves y dónde está)
* Locaclización de objetos
* Reconocimiento de caras [openface](https://github.com/cmusatyalab/openface), [face-alignment](https://github.com/1adrianb/face-alignment)
* [Reconocimiento de emociones](https://github.com/oarriaga/face_classification)

#### Generación (dibuja algo nuevo en base a algo)
* [Aplica filtros de estilo a tus imágenes](https://github.com/jcjohnson/neural-style) (Style Networks)
* Pintor de arte
   * [Con ayuda (de bocecto de paint a cuadro)](https://github.com/alexjc/neural-doodle)
   * Sin ayuda (de descripción a caudro)
* [De imagen a imagen (pix2pix)](https://github.com/phillipi/pix2pix)  (GAN)
   * [Dar color a imágenes en blanco y negro](https://github.com/pavelgonchar/colornet)
   * [image-analogies](https://github.com/awentzonline/image-analogies)
* Caras
   * [Editor de parámetros (genero, pelo, gafas, emoción,...)](https://github.com/ajbrock/Neural-Photo-Editor) (GAN)
   * [Reconstrucción en 3D](https://github.com/AaronJackson/vrn)
   * [Quitar gafas](https://blog.insightdatascience.com/isee-removing-eyeglasses-from-faces-using-deep-learning-d4e7d935376f)
* Crea imagenes desde descripciones
* [Aumentar la resolución](https://github.com/david-gpu/srez) (DCGAN)
* Handwriting Generation (de texto a manuscrito)

#### Ideas
* https://nucl.ai/blog/
* [DL computer vision python book](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)

## Música y audio
* Pasa a texto tu voz (speech recognition) [wavenet](https://github.com/ibab/tensorflow-wavenet)
* [Recomendación de música](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484)
* [Componer música](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)
   * [Música clásica](https://github.com/hexahedria/biaxial-rnn-music-composition)
   * [Música jazz](https://deepjazz.io/)
* Crea tu propio Shazam
* [Clasificar sonidos urbanos](https://github.com/aqibsaeed/Urban-Sound-Classification)

## Vídeo
* Reconocimiento de actividades
* [Leer labios](https://github.com/astorfi/lip-reading-deeplearning)
* Suplantar la cara de otro (face2face)
* Intercamiar tu cara con alguien (face swap)

## Conducción autónoma
* [Detectar semáforos](https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62)
* Conducción autonoma dron [vídeo](https://www.youtube.com/watch?v=umRdt3zGgpU)

## Economía
* Predice los precios de las acciones

## Derecho
* Construye un abogado (chatbot)

#### Más info
* https://www.topbots.com/automating-the-law-a-landscape-of-legal-a-i-solutions/
* https://www.legalrobot.com/
    
## Medicina
* [Predecir ataques de corazón](https://github.com/jisaacso/DeepHeart)
* Analizando imágenes clínicas
* [Analizando tu ADN](https://research.googleblog.com/2017/12/deepvariant-highly-accurate-genomes.html)
* Descubriendo medicamentos
* El fin del médico como persona

#### Más info
* [Nvidia medicine](http://www.nvidia.com/object/deep-learning-in-medicine.html)
* [Nvidia healthcare](https://www.nvidia.com/en-us/deep-learning-ai/industries/healthcare/)
* [Github papers imagen medicas](https://github.com/albarqouni/Deep-Learning-for-Medical-Applications)
* [Github projects biology](https://github.com/hussius/deeplearning-biology)
 
---

# Proyectos usando apis
* [Stanford NLP API](https://stanfordnlp.github.io/CoreNLP/)

---

## Ideas

#### Medium
 * [TensorFlow for Hackers](https://medium.com/@curiousily)
 * [Machine learning is fun](https://medium.com/@ageitgey)

#### Otros
* https://github.com/sachinruk/deepschool.io
* http://deeplearninggallery.com/
* https://ml-showcase.com/
* https://algorithmia.com/algorithms
* https://www.analyticsvidhya.com/blog/2017/02/6-deep-learning-applications-beginner-python/
