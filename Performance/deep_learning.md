# Deep learning PRO


## 1. Read data from disk fast

- ERROR: Slow reading
  - PIL
  - OpenCV
- GOOD: Fast file reading
  - turbojpeg: For faster
- VERY GOOD: Preparsear los ficheros y cargar diercamente el tensor
  - Numpy Memmap or Torch storage
  - [FFCV](https://ffcv.io)

> Note: Buy NVMe drives and put the dataset there


## 2. Fast DataAug & transfoms

- ERROR
  - Torchvision tranforms (based on PIL)
  - Albumatations (based on OpenCV)
- GOOD
  - Fastai agmentations (on GPU)
  - Kornia: agmentations (on GPU)

## 3. Model optimizaation

- Quantization aware training
- Float16

## 4. Training loop optimization
- Composer


## 5. Distributed treainng (Several GPUs)

- [Pytorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html): Data parallelisom
- [DeepSpeed](https://www.deepspeed.ai): : Model parallelisom
- [FSDP](https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html)


> ## References
> - https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/
> - https://towardsdatascience.com/setting-a-strong-deep-learning-baseline-in-minutes-with-pytorch-c0dfe41f7d7
> - https://towardsdatascience.com/gpus-are-fast-datasets-are-your-bottleneck-e5ac9bf2ad27
> - https://towardsdatascience.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719
> - https://neptune.ai/blog/image-segmentation-tips-and-tricks-from-kaggle-competitions







## AVOID CUDA OUT OF MEMORY (OOM) ERROR. 5 + 1 tricks

If you get a OOM error on the first batch, try...

| Trick                     | Description                                      | Disadvantage 
|---------------------------|--------------------------------------------------|--------------------------------------------|
| 1. Smaller batch size     | Use Gradient Accumulation to simultate larger BS | Slower training                            |
| 2. Smaller input size     | Decrease image resolution                        | Lower accuracy                             |
| 3. Smaller model          |                                                  | Lower accuracy                             |
| 4. Half/Mixed precision   | Float16 instead of Float32                       | Lower accuracy. Training may not converge… | 
| 5. Gradient checkpointing | Compute some acts again instad of store them     | Slower training                            |

> Gradient checkpointing is also known as Buffer checkpointing
> Gradient checkpointing reduces the model's memory cost by 60%..70% (at the cost of 25% greater training time).


### Extra 6th trick: Do not store the unnecesary cuda variables in the training loop.

If you get a OOM error in the middle of training, you probably has an increasingly memeory leak.

Delete it after the loss computation
```python
output = model(input)
loss   = loss_fn(output, target)
del output
gc.collect()
```

Or do comoute the loss directly:
```python
loss = loss_fn(model(input), target)
```




| Operation          | Pytorch code         | GPU memory                       | 
|--------------------|----------------------|----------------------------------|
| (0) Model to GPU   | `model.cuda()`       | model                            |
|                    |                      |                                  | 
| (1) Foward pass    | `out = model(inp)`   | model + activations + out        |
| (1) Loss func      | `l = loss(out, tar)` | model + activations + out + l    |
| (1) Backward pass  | `l.backward()`       | model + out + l + grads          |
| (1) Optimizer step | `optimizer.step()`   | model + out + l + grads + grads_mom |
|                    |                      |                                  | 
| (2) Foward pass    | `out = model(inp)`   | model + activations + out        |
| (2) Loss func      | `l = loss(out, tar)` | model + activations + out + l    |
| (2) Backward pass  | `l.backward()`       | model + out + l + grads          |
| (2) Optimizer step | `optimizer.step()`   | model + out + l + grads + grads_mom |


### FAQ

#### Why do I need to store the activations during the foward pass?

Each layer with learnable parameters will need to store its input until the backward pass due to the **chain rule**


> ### Reference
> - [Gradient Accumulation](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html)
> - [Gradient Checkpointing 1](https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs)
> - [Gradient Checkpointing 2](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)
> - [Pytorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
> - [7 Tips To Maximize PyTorch Performance](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)
> - [Memory usage in Pytorch 1](https://medium.com/deep-learning-for-protein-design/a-comprehensive-guide-to-memory-usage-in-pytorch-b9b7c78031d3)
> - [Memory usage in Pytorch 2](https://www.sicara.fr/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks)




## Layer-Wise Learning Rate in PyTorch

https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2022/03/29/discriminative-lr.html
