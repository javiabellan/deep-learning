# Teoría

## Ruta de aprendizaje

1. Redes neuronales
2. Redes neuronales convolucionales
3. Redes neuronales recurrentes
4. GANs
5. Deep Reinforcement Learning


## Sequecias

#### Sequence Prediction

Sequence prediction involves predicting the next value for a given input sequence.
Sequence prediction attempts to predict elements of a sequence on the basis of the preceding elements.

> [1, 2, 3, 4, 5] -> 6

* Weather Forecasting. Given a sequence of observations about the weather over time, predict the expected weather tomorrow.
* Stock Market Prediction. Given a sequence of movements of a security over time, predict the next movement of the security.
* Product Recommendation. Given a sequence of past purchases of a customer, predict the next purchase of a customer.
* Predictor de texto (teclado)

#### Sequence Classification

Sequence classification involves predicting a class label for a given input sequence.

> [1, 2, 3, 4, 5] -> “good” or “bad”

* DNA Sequence Classification. Given a DNA sequence of ACGT values, predict whether the sequence codes for a coding or non-coding region.
* Anomaly Detection. Given a sequence of observations, predict whether the sequence is anomalous or not.
* Sentiment Analysis. Given a sequence of text such as a review or a tweet, predict whether sentiment of the text is positive or negative.

#### Sequence Generation

Sequence generation involves generating a new output sequence that has the same general characteristics as other sequences in the corpus.

> [1, 3, 5], [7, 9, 11] -> [3, 5 ,7]

* Text Generation. Given a corpus of text, such as the works of Shakespeare, generate new sentences or paragraphs of text that read like Shakespeare.
* Handwriting Prediction. Given a corpus of handwriting examples, generate handwriting for new phrases that has the properties of handwriting in the corpus.
* Music Generation. Given a corpus of examples of music, generate new musical pieces that have the properties of the corpus.
* Image Caption Generation. Given an image as input, generate a sequence of words that describe an image.

#### Sequence-to-Sequence Prediction

Sequence-to-sequence prediction involves predicting an output sequence given an input sequence.

> [1, 2, 3, 4, 5] -> [6, 7, 8, 9, 10]

* Multi-Step Time Series Forecasting. Given a time series of observations, predict a sequence of observations for a range of future time steps.
* Text Summarization. Given a document of text, predict a shorter sequence of text that describes the salient parts of the source document.
* Program Execution. Given the textual description program or mathematical equation, predict the sequence of characters that describes the correct output.

#### [Más info](https://machinelearningmastery.com/sequence-prediction/)

## Dimesionalitiy reduction

* Autoencoder: Codificador + Descodificador (comprimir información)
* Restricted boltzmann machine: Como el autoencoder pero va y vuelve
* PCA: Codificador. Reduce el numero de dimensiones
* word2vect: Similar al autoencoder





## Fuentes


### Matemáticas

* [Álgebra lineal](https://www.khanacademy.org/math/linear-algebra)
* [Calculus multivariable](https://www.khanacademy.org/math/multivariable-calculus)
