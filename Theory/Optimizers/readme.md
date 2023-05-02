## Optimizers
- http://dev.fast.ai/optimizer
- https://github.com/jettify/pytorch-optimizer
- https://github.com/lessw2020/Best-Deep-Learning-Optimizers

|                        | Description                                        | Paper                                    | Fast.ai 2                | Score |
|:-----------------------|:---------------------------------------------------|------------------------------------------|--------------------------|-------|
| **SGD**                | Basic method. `new_w = w - lr * grad_w`            |                                          | SGD(lr=0.1)              |       |
| **SGD with Momentum**  | Speed it up with momentum, usually `mom=0.9`       |                                          | SGD(lr=0.1, mom=0.9)     |       |
| **AdaGrad**            | Adaptative lr                                      | 2011                                     | -                        |       |
| **RMSProp**            | Similar to momentum but with the gradient squared. | 2012                                     | RMSProp(lr=0.1)          |       |
| **Adam**               | Momentum + RMSProp.                                | [2014](https://arxiv.org/abs/1412.6980)  | Adam(lr=0.1, wd=0)       | ⭐     |
| **LARS**               | Compute lr for each layer with a certain trust.    | [2017](https://arxiv.org/abs/1708.03888) | Larc(lr=0.1, clip=False) |       |
| **LARC**               | Original LARS clipped to be always less than lr    |                                          | Larc(lr=0.1, clip=True)  |       |
| **AdamW**              | Adam + decoupled weight decay                      | [2017](https://arxiv.org/abs/1711.05101) |                          |       |
| **AMSGrad**            | Worse than Adam in practice. (AdamX: new verion)   | [2018](https://arxiv.org/abs/1904.09237) |                          |       |
| **QHAdam**             | Quasi-Hyperbolic Adam                              | [2018](https://arxiv.org/abs/1810.06801) | QHAdam(lr=0.1)           |       |
| **LAMB**               | LARC with Adam                                     | [2019](https://arxiv.org/abs/1904.00962) | Lamb(lr=0.1)             |       |
| **NovoGrad**           | .                                                  | [2019](https://arxiv.org/abs/1905.11286) |                          |       |
| **Lookahead**          | Stabilizes training at the rest of training.       | [2019](https://arxiv.org/abs/1907.08610) | Lookahead(SGD(lr=0.1))   |       |
| **RAdam**              | Rectified Adam. Stabilizes training at the start.  | [2019](https://arxiv.org/abs/1908.03265) | RAdam(lr=0.1)            |       |
| **Ranger**             | RAdam + Lookahead.                                 | 2019                                     | ranger()                 | ⭐⭐⭐  |
| **RangerLars**         | RAdam + Lookahead + LARS. (aka Over9000)           | 2019                                     |                          | ⭐⭐⭐  |
| **Ralamb**             | RAdam + LARS.                                      | 2019                                     |                          |       |
| **Selective-Backprop** | Faster training by focusing on the biggest losers. | [2019](https://arxiv.org/abs/1910.00762) |                          |       |
| **DiffGrad**           | Solves Adam’s "overshoot" issue                    | [2019](https://arxiv.org/abs/1909.11015) |                          |       |
| **AdaMod**             | Optimizer with memory                              | [2019](https://arxiv.org/abs/1910.12249) |                          |       |
| **DeepMemory**         | DiffGrad + AdaMod                                  |                                          |                          |       |

<p align="center"><img width="66%" src="img/SGD vs Adam.png" /></p>


- **SGD**: `new_w = w - lr[gradient_w]`
- **SGD with Momentum**: Usually `mom=0.9`.
  - `mom=0.9`, means a `10%` is the normal derivative and a `90%` is the same direction I went last time.
  - `new_w = w - lr[(0.1 * gradient_w)  +  (0.9 * w)]`
  - Other common values are `0.5`, `0.7` and `0.99`.
- **RMSProp** (Adaptative lr) From 2012. Similar to momentum but with the gradient squared.
  - `new_w = w - lr * gradient_w / [(0.1 * gradient_w²)  +  (0.9 * w)]`
  - If the gradient in not so volatile, take grater steps. Otherwise, take smaller steps.
- [DiffGrad](https://medium.com/@lessw/meet-diffgrad-new-deep-learning-optimizer-that-solves-adams-overshoot-issue-ec63e28e01b2)
- [AdaMod](https://medium.com/@lessw/meet-adamod-a-new-deep-learning-optimizer-with-memory-f01e831b80bd)

### Optimizers in Fast.ai

You can build every optimizer by doing 2 things:
1. **Stats**: keep track of whats is going on on the parameters
2. **Steppers**: Figure out how to update the parameters

<p align="center"><img width="66%" src="img/optimizers-fastai.png" /></p>


> TODO: Read:
> - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (1998, Yann LeCun)
> - LR finder
>   - [blog](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
>   - [paper](https://arxiv.org/abs/1506.01186)
> - [Superconvergence](https://arxiv.org/abs/1708.07120)
> - [A disciplined approach to neural network hyper-parameters](https://arxiv.org/pdf/1803.09820.pdf) (2018, Leslie Smith)
> - [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)