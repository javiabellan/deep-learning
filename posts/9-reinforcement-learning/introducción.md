# Reinforcement learning

¿Cómo aprende un perro trucos nuevos como sentarse o tumbarse? Cada vez que lo hace bien, se le da un premio. Esa chuche simplemente es un refuerzo positivo. A la larga, nuestra mascota aprenderá que hacer el truco bien tiene una recompensa. La idea se puede extender a los algoritmos que aprenden de forma automática. Tenemos que dar a nuestros algoritmos chuches digitales.


### Entidades
* **Agente**: Algorítmo de aprendizaje
* **Entorno**: El universo donde se encuentra el agente

![RL](https://i.stack.imgur.com/eoeSq.png)

### Ciclo de vida
* **Acción (A)**: La decisión que toma el agente.
* **Estado (S)**: Situación del agente.
* **Recompensa (R)**: feedback inmediato cuande el agente produce una acción.
Cuando el agente genera una nueva acción, pasa a un nuevo estado y además recive una recompensa 

### Estrategias y objetivos
* **Política (π)**: Estrategia que sigue el agente para generar una nueva acción en función de su estado.
* **Valor (V)**: Resultado final esperado. Valor del estado final (S).
* **Valor-Q (Q)**: Similar al Valor, excepto que también tiene en cuenta las acciones.

## Model-free vs Model-based
Los algorítmos basados en modelo aprenden todas las probabilidades de pasar al siguiente estado `P(s1|(s0, a))`.
Sin embargo este método resulta poco práctico ya que las probabilidades a aprender aumentan exponencialmente conforme queremos calcular estados más lejanos (s2, s3, s4...)

Por otra parte, los algorítmos de modelo libre no calculan todas las transiciones. Sio que se basan en parender por prueba y error.

## On-policy v.s. Off-policy

Los agentes con política aprenden el valor con la acción acutal `A`. Mientras, los agentes sin política aprenden en base a `A*`.

An on-policy agent learns the value based on its current action a, whereas its off-policy counter part learns it based on the greedy action a*.

## Q-Learning
Es un algorítmo sin política y sin modelo, basado en en la ecuación de Bellman

## SARSA

## DQN
Ver https://github.com/keon/deep-q-learning

## DDPG

## Deep Neuroevolution
Uber has released [a suite of papers](http://eng.uber.com/deep-neuroevolution/) detailing the use of neuroevolution for training deep neural networks on reinforcement learning tasks. Neuroevolution uses genetic algorithms to train networks, seeing better results than common models like deep Q-learning and A3C.

## Comparación

| Algorítmo      | Modelo     | Política   | Acciones  | Observaciones | Operador |
|:--------------:|:----------:|:----------:|:---------:|:-------------:|:--------:|
| **Q-learning** | Model-free | Off-policy | Discretas | Discretas     | Q-values |
| **SARSA**      | Model-free | On-policy  | Discretas | Discretas     | Q-values |
| **DQN**        | Model-free | Off-policy | Discretas | Continuas     | Q-values |
| **DDPG**       | Model-free | Off-policy | Continuas | Continuas     | Q-values |
| **TRPO**       | Model-free | Off-policy | Continuas | Continuas     | Ventaja  |
| **PPO**        | Model-free | Off-policy | Continuas | Continuas     | Ventaja  |


## Referencias

* [Curso medium de Arthur Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
* [Demystifying deep reinforcement learning](http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/)
* https://hackernoon.com/david-silver-rl-course-lecture-1-notes-a42cd1c6f687
* [Introduction to Various Reinforcement Learning Algorithms Part 1 (Medium)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)
* [Introduction to Various Reinforcement Learning Algorithms Part 2 (Medium)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-part-ii-trpo-ppo-87f2c5919bb9)
* http://www.cse.unsw.edu.au/~cs9417ml/RL1/index.html
* http://www.cse.unsw.edu.au/~cs9417ml/


