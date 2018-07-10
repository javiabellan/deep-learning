


## Pretrained word vectors (2013)

Since then, the standard way of conducting NLP projects has largely remained unchanged:
word embeddings pretrained on large amounts of unlabeled data via algorithms such as word2vec and GloVe
are used to initialize the first layer of a neural network, the rest of which is then trained on data of a particular task

Sometimes the architecture was a [CNN](https://arxiv.org/pdf/1408.5882.pdf).

they have a major limitation: they only incorporate previous knowledge in the first layer of the model,
the rest of the network still needs to be trained from scratch.

* [Word2vec](https://arxiv.org/pdf/1310.4546.pdf): Distributed Representations of Words and Phrases
and their Compositionality
* [GloVe](https://nlp.stanford.edu/pubs/glove.pdf): Global Vectors for Word Representation

![img](https://thegradient.pub/content/images/2018/07/image_0.png)

Using word embeddings is like initializing a computer vision model with pretrained representations that only encode edges:
they will be helpful for many tasks, but they fail to capture higher-level information that might be even more useful.
A model initialized with word embeddings needs to learn from scratch not only to disambiguate words,
but also to derive meaning from a sequence of words.
 
## NLP models
 
At the core of the recent advances of ULMFiT, ELMo, and the OpenAI transformer is one key paradigm shift:
going from just initializing the first layer of our models to pretraining the entire model with hierarchical representations.
If learning word vectors is like only learning edges,
these approaches are like learning the full hierarchy of features, from edges to shapes to high-level semantic concepts.

ULMFiT, ELMo, and the OpenAI transformer have now brought the NLP community close to having an "ImageNet for language"
that is, a task that enables models to learn higher-level nuances of language,
similarly to how ImageNet has enabled training of CV models that learn general-purpose features of images.

* [ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)
* [ELMo](https://arxiv.org/pdf/1802.05365.pdf)
* [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

## NLP applications & datasets
* **Text classification**: ??
* **Question answering/Reading comprehension**: Texts with its question-answer pairs. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
* **Natural language inference**: Sentence pairs with the labels *entailment*, *contradiction*, and *neutral*. [SNLI](https://nlp.stanford.edu/projects/snli/)
* **Coreference Resolution**: [coref](https://nlp.stanford.edu/projects/coref.shtml)
* **Translation**: Same texts in 2 or more languages.
* **Syntax and grammar**: Syntactic structure of sentences in the form of parse trees.
* **Language modeling**: Predict the next word given its previous word. Unlabeled texts (unsupervised).
* **Sentiment analysis**:
* **Constituency parsing**:
* **Skip-thoughts**:
* **Autoencoding**:
* **Named Entity Recognizer (NER)**: Sotware that label  labels sequences of words [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.html)
* **Semantic role labeling (SLR)**

If we ara looking for a analogy for Imagenet, the more difficult question thus is:
Which task is most representative of the space of NLP problems? In other words,
which task allows us to learn most of the knowledge or relations required for understanding natural language?

Language modelling has been shown to capture many facets of language relevant for downstream tasks,
such as long-term dependencies, hierarchical relations, and sentiment

the biggest benefits of language modelling is that training data comes for free with any text corpus
and that potentially unlimited amounts of training data are available.
This is particularly significant, as NLP deals not only with the English language.
More than 4,500 languages are spoken around the world by more than 1,000 speakers.

Language modeling as a pretraining task has been purely conceptual.
But in recent months, we also obtained empirical proof
Embeddings from Language Models (ELMo),
Universal Language Model Fine-tuning (ULMFiT),
and the OpenAI Transformer
have empirically demonstrated how language modeling can be used for pretraining
They achieve state-of-the-art on a diverse range of tasks in Natural Language Processing.


## Tecnologies
* Conditional Random Field (CRF)
* bi-LSTM

## References

* [NLP Imagenet moment has arrived](https://thegradient.pub/nlp-imagenet/)
