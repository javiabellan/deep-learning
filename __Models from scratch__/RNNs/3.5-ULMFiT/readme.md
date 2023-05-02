## Language model (ULMFiT)

* Data
  * LM pretraining
    * General-domain corpus. Should be large and capture general properties of language.
    * Wikitext-103: consisting of 28,595 preprocessed Wikipedia articles and 103 million words.
    * We leave the exploration of more diverse pretraining corpora to future work
  * LM finetuning
    * Target task corpus
  * Pre-processing
    * We add special tokens for upper-case words, elongation, and repetition
* Model
  * Embedding size of 400 (input)
  * 3-layer LSTM + dropout ([AWD-LSTM](https://arxiv.org/pdf/1708.02182.pdf))
    * a regular LSTM with no attention, short-cut connections, or other sophisticated additions.
    * dropout of 0.4
    * 1150 hidden activations per layer
  * NOTE: Analogous to CV, we expect better language models in the future.
  * Classifier fine-tuning: Augment with two additional linear blocks.
* Training (for finetuning)
  * Gradual unfreezing: Starting from the last layer, then unfreeze the next...
  * Discriminative fine-tuning: different learning for each layer
  * Slanted triangular learning rate (STLR)
  * Adam with β1=0.7 (instead of the default β1=0.9) and β2=0.99
  * Backpropagation through time (BPTT)
    * To enable gradient propagation for large input sequences.
    * Divide the document into fixed-length batches.
    * At the beginning of each batch, the model is initialized with the final state of the previous batch.


![img](https://thegradient.pub/content/images/2018/07/image_9.png)

#### Other tecnologies
* Conditional Random Field (CRF)
* bi-LSTM
