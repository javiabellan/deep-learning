# Redes neuronales

Las redes neuronales artificiales, (también llamadas perceptrones multicapa) son el fundamento del deep learning.
Y como el deep learning es el campo más prometedor de la inteligencia artificial a día de hoy,
podría decirse que las redes neuronales son el fundamento de inteligencia artificial moderna.

### Definición
El objetivo de una red neuronal es **aproximar una función**, cualquier función.
Esto quiere decir que una red neuronal es capaz de transformar unos datos de entrada en unos datos salida,
al igual que lo haría una función matemática.
Se usa el término aproximar porque normalmente la función que se quere conseguir se desconoce,
y el objetivo de la red neuronal es acercarse lo maximo posible y conseguir esa función
partiendo solamete de los datos de entrada con su correspondiente salida.

<p align="center"><img src ="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Function_machine2.svg/220px-Function_machine2.svg.png" /></p>

Exiten funciones simples como la suma, donde los datos de entrada son los números a sumar, y la salida es la suma.
Pero también existen funciones complejas como el reconocimiento de imágenes, donde la entrada son los pixels de la imagen,
y la salida es la clase del objeto que se detecta. Una red neuronal será capaz de aproximar cualquier función
si tiene la estructura adecuada y le damos suficientes datos para que aprenda, veamos como.

### Funcionamiento

La red neuronal se compone por varias unidades de procesamiento contectadas unas con otras llamadas perceptrones (o coloquialmente neuronas).
Un perceptrón podría definirse como la implemeteación matemática de una neurona biológica.

Una **neurona** produce una señal eléctrica por su salida (axon), cuando recive  la suficiente carga electrica por sus entradas (dentritas).
La salida de una neurona se conecta a la entrada de otra mediante una sinpasis.
Cuanto mayor sea la sinapsis, más probabilidad de que la segunda neurona se active cuando se active la primera.

<p align="center"><img width="50%" src ="http://web.cs.ucla.edu/~forns/assets/images/spring-2014/cs-161/week-10/neural-1.PNG" /></p>

El **perceptrón** imita dicho funcionamiento. Las señales eléctricas se convierten en valores numéricos.
La entrada `x`, es una lista de números (vector) que se multiplica por los pesos sinápticos `w` (weights en inglés) del perceptrón.
Estas multiplicaciones se suman y se aplica una función de acctivación para producir la salida.
Para entender con mayor detalle como funciona, visita la página del [perceptrón](/teoría/modelos/perceptron.md).

<p align="center"><img src=https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Perceptr%C3%B3n_5_unidades.svg/400px-Perceptr%C3%B3n_5_unidades.svg.png /></p>

Una **red neuronal** se compone por perceptrones distribuidos por capas.
Cada perceptrón, toma como entrada todas las salidas de la capa anterior,
para producir su propia salida que, será transmitida a los perceptrones de la capa siguiente.
La primera capa (input layer) son los datos, no son perceptrones. 

<p align="center"><img src=http://www.cs.us.es/~fsancho/images/2017-02/neuralnetwork.png /></p>

El término *deep* (profundo) de deep learning hace referencia al número de capas de la red.
Normalmente, con mayor profundidad (más capas) se pueden extraer características más abstractas de los datos.
Más adelante veremos como elegir un número adecuado de capas y de perceptrones por capa.
Pero teóricamente podemos aproximar cualquier función con tan sólo una capa oculta con las suficientes neuronas.
