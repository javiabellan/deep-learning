# Q-learning

Qeremos conseguir algo, ganar un juego. Como hemos dicho, aprendizaje por refurezo se basa en dar recomensas cuando el algoritmo lo hace bien.
Pero tenemos que distinguir varios tipos de recomesas. Por ejemplo en el juego de ajedrez:

* Recompensa a largo plazo: Ganar la partida
* Recompenensa a corto plazo: Comer fichas

## Datos

El algorimo, necesita 2 cosas:
* Recompesas: Saber cuando se ha ganado o se ha perdido
* Experiencias: Jugar mucho para saber como se gana o se pierde.


### Recompensa = r(estado, acción)

Peor ejemplo, en el 3 en raya, el estado es la configuración del tablero, y la acción es nuestra jugada.

![](https://rubenlopezg.files.wordpress.com/2015/05/direct_reward_int.png)

Si no ganamos, la recompensa es 0.

![](https://rubenlopezg.files.wordpress.com/2015/05/direct_reward_win.png)

Si ganamos, la recompensa es 1.

La función de recompesa se puede hacer:
* Con una tabla que tenga todas las configuraciones del tablero (Solución bruta)
* Analizando el estado de alguna forma (Solución mas viable)

### Experiencia = (estado, acción, recompensa, nuevo estado)

![](https://rubenlopezg.files.wordpress.com/2015/05/experience1.png)

Fijarse que el nuevo estado, es la respueta del entorno a nuestra acción.
En el caso de las 3 en raya, el nuevo estado contedrá el movimiento que hace el oponente.

## Algorítmo

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

## Q-learning en sistemas continuos

Juego como el el 3 en raya o el ajedrez, tiene un conjunto discreto de posibilidades a realizar.
Pero existen otros problemas, que implican tomar una acción en un rango continuo,
como por ejemplo un robot aspiradora, o el control de un automóvil.


---

 Ademas el algoritmo tiene 2 parametros:

* Velocidad de aprendizaje (learning rate). Es un valor entre 0 y 1 que indica cuánto podemos aprender de cada experiencia. 0 significa que no aprendemos nada de una nueva experiencia, y 1 significa que olvidamos todo lo que sabíamos hasta ahora y nos fiamos completamente de la nueva experiencia.
* Factor de descuento (discount factor). Es también un valor entre 0 y 1 que indica cuán importante es el largo plazo. 0 significa que sólo nos importan los refuerzos inmediatos, y 1 significa que los refuerzos inmediatos no importan, sólo importa el largo plazo. Ojo, porque valores muy cercanos a 1 tienden a divergir. Este factor nos ayuda a mezclar recompensas directas con recompensas a largo plazo y producir la recompensa mixta.

---

## Referencias

* https://rubenlopezg.wordpress.com/2015/05/12/q-learning-aprendizaje-automatico-por-refuerzo/
* [Modelos ocultos de Markov](http://artint.info/html/ArtInt_161.html)
* [Diferencias temporales](http://artint.info/html/ArtInt_264.html)


