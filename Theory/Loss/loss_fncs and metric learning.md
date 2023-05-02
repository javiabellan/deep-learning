# Metric Learning (Similarity learning)

Training a similarity function that measures how similar or related two objects are). Cross-entropy is valid choice of loss function, same as contrastive or triplet on L2 distance

Classification:
	Embedding from different classes need to be easily separable.
Metric learning:
	Embedding from the same class need to be close together, and embedding from different classes need to be far from each others.

cosine similarity, euclidean

Other names:
- Metric learning
- Embedding learning
- Similarity learning
- Siamese network


## Mining
Mining is the process of finding the best pairs or triplets to train on. T

## Losses

### Pairwise Losses (Parejas de datos de entrada)

- Concat embs                 -> Binary classification (fastai siamese)
  - Downside: The model is not "symmetrical", (the input of [img A, img B] will give a different output from [img B, img A]
- Absolute difference of embs -> Binary classification
- Cosine similarity of embs   -> Regression


### Triplet Losses (Trios como datos de entrada)
- Triplet Loss
  1. Anchor: represents a reference
  2. Positive: same class as the anchor
  3. Negative: a different class

Contrastive Loss
Margin Loss
Hinge Loss

Contrastive
ArcFace, CosFace, SphereFace 



######################### CLASIFICATION

Losses:
- Categorical Cross-Entropy Loss,
- Binary Cross-Entropy Loss,
- Softmax Loss,
- Logistic Loss,
- Focal Loss


(5/n) Metric learning have won Kaggle classification competitions of unbalanced data:

- https://kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
- https://www.kaggle.com/c/humpback-whale-identification/discussion/82366