---
Autor: Javi
---
# Predicción de palabras

Vamos a predir la palabra en base a las palabras anteriores. Este predictor se puede incorporar a un teclado para poder predecir las palabras siguientes. El contenido que se va a ver es el siguiente: 

* Repaso de teoría
  * Redes neuronales recurrentes (LSTM)
  * Word embeddings
  * Encoder-Decoder
* Implementación en Tensorflow
  * Limpiar datos
  * Construir vocabulario
  * Convertir palabras a vectores
  * Definir el modelo (Encoder-Decoder)
  * Entrenar el modelo
  * Probar el modelo

## Redes neuronales recurrentes (LSTM)

Usaremos una red recurrente pq es capaz de memorizar cosas.

La entrada de una neurona RNN, es la entrada normal + la salida suya de antes:

`salidaActual = act( (entradaActual + salidaAnterior) * pesos)`

El problea de las RNN es que el problema de **vanishing and exploding gradient** les afecta mucho debido a su recursividad. Por eso usaremos untipo especial de RNN: las **LSTM**, que guradan un estado en su interior. Las LSTM superan el problema del vanishinh/exploding gradient al mantener el error que debe ser propagado hacia atrás en el agoritmo de backpropagation.

## Word embeddings

## Encoder-Decoder

## Datos

Vamo a usar el libro *Más allá del bien y del mal* de Friedrich Nietzsche para entrenar nuestra red. Y pasamos todas las letras a minúscula para simplificar.

```python
ruta = 'nietzsche.txt'
texto = open(ruta).read().lower()
print('Número de letras:', len(text))
```

Vemos que tenemos un texto de 600893 letras para entrenar la red

## Modelo

Concretamente vamos a pasarle un cadena de **40** letras y la red tendrá que predecir la siginte letra,


# Implementación en Tensorflow
Vamos a ver como se hece un

## Descargar datos
Lo primero que tenemos que hacer es conseguir unos buenos datos, vamos a descargar un libro de Shakespeare.

```python
data_dir = 'data'
data_file = 'shakespeare.txt'
if not os.path.isfile(os.path.join(data_dir, data_file)):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('Data file not found, downloading the dataset')
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
```

## Limpiar datos
Vamos a limpar los datos, quitar signos de puntuación raros y todo eso.

```python
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()
```

## Construir vocabulario

```python
def build_vocab(text, min_word_freq):
    word_counts = collections.Counter(text.split(' '))
    print ('word counts: ', len(word_counts), 'text len: ', len(text.split(' ')))
    # limit word counts to those more frequent than cutoff
    word_counts = {key: val for key, val in word_counts.items() if val > min_word_freq}
    # Create vocab --> index mapping
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (ix + 1) for ix, key in enumerate(words)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}
    return (ix_to_vocab_dict, vocab_to_ix_dict)

# Build Shakespeare vocabulary
min_word_freq = 5  # Trim the less frequent words off
ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert (len(ix2vocab) == len(vocab2ix))
```

## Convertir vocabulario a vectores de palabras

```python
s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)
```

## Construir el modelo

```python
# TODO
```

## Entrenar el modelo

```python
# TODO
```

## Probar el modelo

```python
# TODO
```

## Otras arquitecturas a considerar:

* Método tradicional: Wikipedia: Markov chain, y, language model
* Método DL: https://arxiv.org/pdf/1709.06429.pdf
* CNN: 20 * conv layer with kernel size=5, dimensions=300

## Referencias
* [Predecir solo letra](link: https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218)
* [Predecir palabra](https://github.com/dipendra009/Text-generation)
