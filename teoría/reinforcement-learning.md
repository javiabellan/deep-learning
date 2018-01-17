# Reinforcement learning


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

## Modelo, SÍ o NO

## Política, SÍ o NO

Los agentes con política
An on-policy agent learns the value based on its current action a, whereas its off-policy counter part learns it based on the greedy action a*. (We will talk more on that in Q-learning and SARSA)

## Q-Learning

## SARSA

## DQN

## DDPG

## Referencias

* [Introduction to Various Reinforcement Learning Algorithms (Medium)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)
