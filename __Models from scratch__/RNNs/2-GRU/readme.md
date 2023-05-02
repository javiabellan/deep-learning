# GRU

<p align="center">
  <img width="70%" src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png" />
</p>

Tiene dos activaciones:
* **r (combinar)**: Determina cómo combinar la nueva entrada con la memoria anterior.
* **z (actualizar)**: Defina cuanta, de la memoria anterior, mantener.

If we set the reset to all 1’s and  update gate to all 0’s we again arrive at our plain RNN model. The basic idea of using a gating mechanism to learn long-term dependencies is the same as in a LSTM, but there are a few key differences:

- [GRU paper](https://arxiv.org/pdf/1412.3555.pdf)
