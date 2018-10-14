# Reinforcement learning

## Algorítmos

| Algoritmo  |                  Descripcción                  |      Familia     |                                                                     Explicación                                                                     |                             Paper                            |
|------------|:----------------------------------------------:|:----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------:|
| Q-learning |                                                | Q-learning       |                                                                                                                                                     |                                                              |
| SARSA      |                                                | Q-learning       |                                                                                                                                                     |                                                              |
| DQN        |                                                | Q-learning       |                                                                                                                                                     |                                                              |
| DDPG       |                                                | Q-learning       |                                                                                                                                                     |                                                              |
| PGQ        | Combinación de Q-learning y policy gradients   | QL + PG          |                                                                                                                                                     | [Paper](https://arxiv.org/abs/1611.01626)                    |
| A3C        | Asynchronous Advantage Actor-Critic            | Policy gradients | [Medium](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2) | [Paper](https://arxiv.org/abs/1602.01783)                    |
| TRPO       | Trust Region Policy Optimization               | Ventaja          |                                                                                                                                                     | [Paper](https://arxiv.org/abs/1502.05477)                    |
| PPO        |                                                | Ventaja          |                                                                                                                                                     |                                                              |
| ES         | Evolutionary Strategy                          | Genético         | [openAI](https://blog.openai.com/evolution-strategies/)                                                                                             | [Paper](https://arxiv.org/abs/1703.03864)                    |
| DQN+CTS    | Count-Based Exploration + Intrinsic Motivation | Q-learning       |                                                                                                                                                     | [Paper](https://arxiv.org/abs/1606.01868)                    |
| CEM        | Cross Entropy method for Fast Policy Search    |                  |                                                                                                                                                     | [Paper](http://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf) |
|            |                                                |                  |                                                                                                                                                     |                                                              |


## Components

* Agent
* Environment 
* state
* action
* reward

The goal of the agent is to maximize the reward



## Monte Carlo vs TD Learning
* Monte Carlo: Reward at the end
* Temporal Difference Learning: Reward at each step


## Exploration/Exploitation trade off
* Exploration is finding more information about the environment.
* Exploitation is exploiting known information to maximize the reward.

## Approaches

#### Value-based
To choose which action to take given a state, we take the action with the highest Q-value (maximum expected future reward I will get at each state).

#### Policy-based
In policy-based methods, instead of learning a value function that tells us what is the expected sum of rewards given a state and an action, we learn directly the policy function that maps state to action

A policy can be either deterministic or stochastic.
* A deterministic policy maps state to actions
* A deterministic policy outputs a probability distribution over actions.

#### Model-based



* Value-based
  * Q-learning (2014, Deepmind)
  * Deep Q-learning (DQN)
  * SARSA
  * DDPG
* Policy-based
  * Policy gradients
  * Advantage Actor Critic (A2C)
  * Asynchronous Advantage Actor Critic (A3C)
  * Proximal Policy Optimization (PPO) (2017, OpenAI)
  * TRPO
C51
Rainbow
Implicit Quantile
Evolutionary Strategy
Genetic Algorithms
