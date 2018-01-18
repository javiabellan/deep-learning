# Q-learning

El objetivo es realizar la acción optima en cada momento, para llegar a un estado final ganador.
Al final se trata de encontrar una **función** `Q(s,a)` que te diga lo bueno que es hacer la **acción** `a` para el **estado** `s`.
Esto te da una probabilidad o valor, (podemos ser fieles, o explorar otras acciones como veremos más adelante).

Podemos representar la función Q en una tabla, que contenga los valores para todos los posibles estados y acciones. Al principio, como no tenemos datos, rellenamos la con valores bajos (por ejemplo 0) pero ponemos valores altos (por ejemplo 1) cuando se trate de un par estado-acción ganadora. Estos valores son las recompensasPor ejemplo, en el juego de 3 en raya sería cuando ganamos.

### Recompensa = r(estado, acción)

Peor ejemplo, en el 3 en raya, el estado es la configuración del tablero, y la acción es nuestra jugada.

![](https://rubenlopezg.files.wordpress.com/2015/05/direct_reward_int.png)

Si no ganamos, la recompensa es 0.

![](https://rubenlopezg.files.wordpress.com/2015/05/direct_reward_win.png)

Si ganamos, la recompensa es 1.

> Pero tenemos que distinguir varios tipos de recomesas. Por ejemplo en el juego de ajedrez:
> * Recompensa a largo plazo: Ganar la partida
> * Recompenensa a corto plazo: Comer fichas

La función de recompesa se puede hacer:
* Con una tabla que tenga todas las configuraciones del tablero (Solución bruta)
* Analizando el estado de alguna forma (Solución mas viable)

![](https://rubenlopezg.files.wordpress.com/2015/05/direct_reward1.png)

## Algorítmo

Una vez tengamos nuestra función Q inicial, necesitamos jugar nuchas partidas (al principio a ciegas)
para poder ir generando experiencais y aprender.
Cuando llegemo a una solución ganadora, significa que nuestras acciones para llegar ahí fueron buenas y por lo tanto debemos aprenderlas

#### Experiencia = (estado, acción, recompensa, nuevo estado)

![](https://rubenlopezg.files.wordpress.com/2015/05/experience1.png)

Una experiencia es simplemente cauando relizamos una nueva acción en un estado determinado,
recivimos una recompensa (positiva o negativa) y pasamos a un nuevo estado.
Fijarse que el nuevo estado, es la respueta del entorno a nuestra acción.
En el caso de las 3 en raya, el nuevo estado contedrá el movimiento que hace el oponente.

Las experiencias son los datos fundamentales para aprender. Pero ojo,
no podemos generar muchos datos para aprender (como el aprendizaje supervisado y no supervisado)
ya que las experiencas necesitan saber la acción que se toma, y al principio no sabemos nada.
Por lo tanto es necesario que el algortimo aprenda en tiempo real (online learning),
y para ello, necesitamos introducir la idea de **exploración**.

Al principio, las estimaciones serán pésimas, pero aunque empiecen a ser correctas y las seguimos a ciegas,
es posible que nunca descubramos mejores jugadas.
Es como el dicho *"arriesgarse a cometer errores es la mejor forma de aprender"*.
Pues bien, cuando no tenemos datos, esto es fundamental.
Si queremos que nuestro conocimiento contenga el gran espacio de posibilidades,
es necesario arriesgarse a perder para explorar opciones nuevas.

La gran pregunta es cómo encontrar un equilibrio razonable entre explorar e ir a lo seguro.
Existen múltiples opciones, pero una bastante común consiste en colocar las diferenteas acciones en una distribución de probabilidad.
Supongamos un sistema con 5 acciones con diferentes probabilidades. En total, las probabilidades suman 0.72.
Si elegimos un número al azar en el rango de la regla (de 0 a 0.72), eligiremos la acción a realizar.

![](https://rubenlopezg.files.wordpress.com/2015/05/exploring2.png)

De esa forma, las acciones con mayor probabilidad serán elegidas más a menudo,
pero todas las acciones serán elegidas en algún momento.

Otra pregunta es hasta cuándo seguir usando exploración. Porque se llegará a un punto en que no se aprende nada nuevo.
En ese caso podemos dejar de explorar, y elegir la mejor acción siempre, puesto que ya hemos aprendido.
Pero existe la posibilidad que el entorno sea dinámico, y la mejor acción cambie con el tiempo.
En este otro caso, puede ser interesante mantener siempre la exploración activa.

> #### Curiosidad
> Un buen método de aprendizaje para juegos de 2 jugadores es que algoritmo juege contra sí mismo.
> De esta forma siempre tendrá un rival de su nivel y mejorará a niveles superiores a los humanos.

> #### Cuidado!
> Hay que controlar el conjunto de acciones posibles que el algoritmo puede hacer.
> Por ejemplo, no se puede poner una ficha encima de otra en el 3 en raya.

### Actualizar la función Q

* α = Learning rate. Valor entre 0 y 1. Cuanto más cercano a 1, más se acutializa.
* γ = Discount factor.  Valor entre 0 y 1 que indica cuán importante es el largo plazo. 0 significa que sólo nos importan los refuerzos inmediatos, y 1 significa que los refuerzos inmediatos no importan, sólo importa el largo plazo. Ojo, porque valores muy cercanos a 1 tienden a divergir. Este factor nos ayuda a mezclar recompensas directas con recompensas a largo plazo y producir la recompensa mixta.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/1df368653bf2eb16081f8738486ef4c9d60e9d03)

El **learned value**, este es el nuevo valor de la función Q para un estado s y acción a determinados,
(gracias al learning rate no olvidamos del todo el antiguo valor Q). Vamos a fijarnos en este learned value.

```python
Q(s,a) = r(s,a) + λ·max(a,)
```

## Q-learning en sistemas continuos

Juego como el el 3 en raya o el ajedrez, tiene un conjunto discreto de posibilidades a realizar.
Pero existen otros problemas, que implican tomar una acción en un rango continuo,
como por ejemplo un robot aspiradora, o el control de un automóvil.


## Referencias

* https://rubenlopezg.wordpress.com/2015/05/12/q-learning-aprendizaje-automatico-por-refuerzo/
* [Modelos ocultos de Markov](http://artint.info/html/ArtInt_161.html)
* [Diferencias temporales](http://artint.info/html/ArtInt_264.html)


