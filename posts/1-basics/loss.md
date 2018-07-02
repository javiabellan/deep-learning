* [Loss functions](/posts/1-basics/loss.md) (Criterium)
  * **Mean Absolute Error** (L1 loss): Regression (for bounding boxes?).
  * **Mean Squared Error** (L2 loss): Regression. Penalice bad misses by too much (for single continuous value?).
  * **Cross Entropy**: Sigle-label classification. Usually with **softmax**.
    * **Negative Log Likelihood** is the one-hot simplified version, see [this](https://jamesmccaffrey.wordpress.com/2016/09/25/log-loss-and-cross-entropy-are-almost-the-same/)
  * **Binary Cross Entropy**:  Multi-label classification. Usually with **sigmoid**.
  
## Focal loss

```python
FL(p) = -(1-p)^gamma * log(p)
```
