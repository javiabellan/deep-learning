## Alphago
Alphago Fan, derrotó al campeón europeo, Fan Hui, en Octubre de 2015. Está formada por dos redes neuronales:

* **Policy network**: Genera las probabilidades de los movimientos: Entrenada inicialamete por movimientos de jugadores expertos y luego por apredendizaje por refuerzo (policy-gradient)
* **Value network**: Evalua la posición. Predece el ganador de la partida. Entrenada con jugando contra sí misma.

Un vez entrenadas, las redes son combinadas con **Monte Carlo tree search** (MCTS) para proporcionar una búsqueda de las posibles jugadas posteriores.

* Usando la **Policy network** para reducir la búsqueda a los movimiento más probables
* Y usando la **Value network** (junto a los Monte Carlo rollouts) para evaluar las posiciones en el árbol de búsqueda.

Una versión posterior, Alphago Lee, derrotó al campeón mundial Lee Sedol, usando métodos similares en marzo de 2016.
 
## Aplhago zero

Alpha zero difiere de las versiones anteriores en difernetes aspecos:

* Es entrenada **solo con aprendizaje por refuerzo**, (sin datos de jugares expertos). Al principio los movimientos son aleatorios y juagando contra sí misma va aprendiendo.
* Sólo usa las fichas blancas y negras como datos de entrada. Ningún dato más.
* Sólo usa una red neuronal.
* Usa un árbol de busqueda más simple sobre esta red para evaluar las posiciones y movimientos. Sin usar Monte Carlo rollouts.

Para ello se introduce un nuevo algoritmo de aprendizaje por refuerzo.
Incluye la búsqueda lookahead dentro del bucle de entrenamiento.
Consiguiendo una mejora rápida y un aprendizaje estable.

## Aprendizaje por refuerzo

Usa un red neuronal profunda *f* con parametros *θ*.

* **Entrada**: Estado del tablero *s*. Es la distribución actual más las jugadas anteriores
* **Salida**: Probabilidad de movimientos y valor. (*p*, *v*).
  * El vector *p*, representa la probabilidad de seleccionar cada movimiento (incluido pasar). `p = Pr(a|s)`
  * El escalar *v*, estima la probabilidad del jugador actual de ganar desde la posición *s*.

Esta red combina las policy y value networks en una sola red. consiste en muchos bloques residuales de capas convolucionales,
con batch normalization y rectifier nonlinearities (Ver métodos)


 
## Referencias
* [Artículo en deep mind](https://deepmind.com/blog/alphago-zero-learning-scratch/)
* [PDF del articulo en Nature](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
* [How to build your own AlphaZero in Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)
* https://medium.com/@amaub/alphazero-and-the-curse-of-human-knowledge-324a3a0ad1b6
* https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef
* [Alphago te enseña a jugar al go](https://alphagoteach.deepmind.com/)
