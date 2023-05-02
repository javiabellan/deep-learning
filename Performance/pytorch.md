

# Pytorch dataloader



## Naive implementatation

```
           Read batch 1 (B1)
          ┌────────────────────────────────────┐                               ┌─────────────────
          │slow|slow|   |slow| slow  |unecesary│  <-- No reads in parallel     │
CPU CORE1 │read|read|...|read|Collate|copy from│  <-- No queue/prefetching     │ Read batch 2 ...
          │img1|img2|   |imgN|       |pag 2 pin│                               │
          └────────────────────────────────────┘                               └─────────────────
                                               ┌─────────┐
CPU 2 GPU                                      │B1 to GPU│ <-- No queue/prefetching on GPU 
                                               └─────────┘
                                                         ┌─────────────────────┐
 GPU                                                     │ INFER B1 into MODEL │ <-- Slow model
                                                         └─────────────────────┘
```

## Read faster

### Fast hardaware

Try to store your dataset in a fast storage

1. RAM
2. SSDs NVMe RAID
3. SSD NVMe
4. SSDs SATA RAID
5. SSD SATA
6. HDD RAID
7. HDD SATA
8. Other machine in local network (/mnt/media mounting point)
9. Internet


### Fast decoding `libjpg-turbo`

- Using [jpeg4py](https://github.com/ajkxyz/jpeg4py): `jpeg.JPEG(img).decode()` instead of `np.array(Image.open(img))`
  - [example of usage](https://www.pankesh.com/posts/2019-05-02-pytorch-augmentation-with-libjpeg-turbo)
- Using `Pillow>=9.0.0` [source](https://pillow.readthedocs.io/en/stable/releasenotes/9.0.0.html#switched-to-libjpeg-turbo-in-macos-and-linux-wheels)

Notese que a partir de la la verion 9 de PIL, viene por defecto 

> Comprobar que PIL usa libjpg-turbo
> ```python
> import PIL.features
> print(PIL.features.check_feature("libjpeg_turbo"))
> ```

### Even faster:

**Precompute tensors and save the on disk**

**For reading just memap, No need to decoding**

- Numpy Memmap
- Torch storage
- [FFCV](https://ffcv.io)

```
           Read batch 1 (B1)
          ┌────────────────────────────────────┐                               ┌─────────────────
          │fast|fast|   |fast| slow  |unecesary│                               │
CPU CORE1 │read|read|...|read|Collate|copy from│                               │ Read batch 2 ...
          │img1|img2|   |imgN|       |pag 2 pin│                               │
          └────────────────────────────────────┘                               └─────────────────
                                               ┌─────────┐
CPU 2 GPU                                      │B1 to GPU│
                                               └─────────┘
                                                         ┌─────────────────────┐
 GPU                                                     │ INFER B1 into MODEL │
                                                         └─────────────────────┘
```

## Fast collate (= concat imags into batch)

https://www.pankesh.com/posts/2019-05-02-pytorch-augmentation-with-libjpeg-turbo/

```
           Read batch 1 (B1)
          ┌─────────────────────────────────┐                                ┌─────────────────
          │fast|fast|   |fast|fast|unecesary│                                │
CPU CORE1 │read|read|...|read|Coll|copy from│                                │ Read batch 2 ...
          │img1|img2|   |imgN|ate |pag 2 pin│                                │
          └─────────────────────────────────┘                                └─────────────────
                                            ┌─────────┐
CPU 2 GPU                                   │B1 to GPU│
                                            └─────────┘
                                                       ┌─────────────────────┐
 GPU                                                   │ INFER B1 into MODEL │
                                                       └─────────────────────┘
```

The `collate_fn` is also useful to discard broken images

```python
# a collate function that filters the None records.
def collate_fn(batch):
    # batch looks like [(x0,y0), (x4,y4), (x2,y2)... ]
    batch = [(Id, Date, Img) for (Id, Date, Img) in batch if Img is not None]
    #batch = list(filter(lambda x: x is not None, batch)) # Other way to do the same
    
    if len(batch) == 0: # If all images are broken, retrun None and discard in dl for loop
        return None, None, None
    else:
        return torch.utils.data.dataloader.default_collate(batch)
```


## Optimization: Avoid unnecessary host copies `pin_memory=True`

Host (CPU) data allocations are pageable by default. The GPU cannot access data directly from pageable host memory, so when a data transfer from pageable host memory to device memory is invoked, the CUDA driver must first allocate a temporary page-locked, or “pinned”, host array, copy the host data to the pinned array, and then transfer the data from the pinned array to device memory, as illustrated below.

- https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

if `pin_memory=True` is set on the Pytorch's dataloader, it will copy Tensors into device/CUDA pinned memory before returning them.

Also, once you pin a tensor or storage, you can use asynchronous GPU copies. Just pass an additional `non_blocking=True` argument to a `to()` or a `cuda()` call. This can be used to overlap data transfers with computation. (Prefetching on GPU Optimization).


```
           Read batch 1 (B1)
          ┌───────────────────────┐                               ┌─────────────────
          │fast|fast|   |fast|fast│                               │
CPU CORE1 │read|read|...|read|Coll│                               │ Read batch 2 ...
          │img1|img2|   |imgN|ate │                               │
          └───────────────────────┘                               └─────────────────
                                  ┌─────────┐
CPU 2 GPU                         │B1 to GPU│
                                  └─────────┘
                                            ┌─────────────────────┐
 GPU                                        │ INFER B1 into MODEL │
                                            └─────────────────────┘
```

## Optimization: Read images in parallel `num_workers`

```
          ┌──────────────┐         ┌──────────────┐
CPU CORE1 │ Read batch 1 │         │ Read batch 4 │
(worker1) └──────────────┘         └──────────────┘
          ┌──────────────┐         ·                       ┌──────────────┐
CPU CORE2 │ Read batch 2 │         ·                       │ Read batch 5 │
(worker2) └──────────────┘         ·                       └──────────────┘
          ┌──────────────┐         ·                       ·                       ┌──────────────┐
CPU CORE3 │ Read batch 3 │         ·                       ·                       │ Read batch 6 │
(worker3) └──────────────┘         ·                       ·                       └──────────────┘
                         ┌─────────┐             ┌─────────┐             ┌─────────┐
CPU 2 GPU                │B1 to GPU│             │B2 to GPU│             │B3 to GPU│
                         └─────────┘             └─────────┘             └─────────┘
                                   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
GPU                                │ INFER MODEL │         │ INFER MODEL │         │ INFER MODEL │
                                   └─────────────┘         └─────────────┘         └─────────────┘
```

> ## Memory optimization
> **Avoid to each worker make a copy of the dataset!!!**
> https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/


## Optimization: Prefetching on CPU (Queue) `prefetch_factor`


**This is the defual behavior of Pytroch's dataloader**. It does prefetching on the CPU RAM.

the prefetch_factor parameter of PyTorch DataLoader class. The prefetch_factor parameter only controls CPU-side loading of the parallel data loader processes


```
          ┌────────────┬────────────┬────────────┐
CPU CORE1 │Read batch 1│Read batch 4│Read batch 7│
          └────────────┴────────────┴────────────┘
          ┌────────────┬────────────┐                          ┌────────────┐
CPU CORE2 │Read batch 2│Read batch 5│                          │Read batch 8│
          └────────────┴────────────┘                          └────────────┘
          ┌────────────┬────────────┐                          ·                             ┌────────
CPU CORE3 │Read batch 3│Read batch 6│                          ·                             │ Read B9
          └────────────┴────────────┘                          ·                             └────────
                       ┌─────────┐                   ┌─────────┐                   ┌─────────┐
CPU 2 GPU              │B1 to GPU│                   │B2 to GPU│                   │B3 to GPU│
                       └─────────┘                   └─────────┘                   └─────────┘
                                 ┌───────────────────┐         ┌───────────────────┐         ┌────────
GPU                              │INFER B1 into MODEL│         │INFER B2 into MODEL│         │INFER B3 
                                 └───────────────────┘         └───────────────────┘         └────────
```

- [But what are PyTorch DataLoaders really?](https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html)
- [Building a Multi-Process Data Loader from Scratch](https://teddykoker.com/2020/12/dataloader)
  - The full code for this project is available at github.com/teddykoker/tinyloader
- https://www.jpatrickpark.com/post/loader_sim/





## Optimization: Prefetching on GPU

```
          ┌────────────┬────────────┐
CPU CORE1 │Read batch 1│Read batch 4│
          └────────────┴────────────┘
          ┌────────────┬────────────┐
CPU CORE2 │Read batch 2│Read batch 5│
          └────────────┴────────────┘
          ┌────────────┬────────────┐
CPU CORE3 │Read batch 3│Read batch 6│
          └────────────┴────────────┘
                       ┌─────────┬─────────┐         ┌─────────┐         ┌─────────┐
CPU 2 GPU              │B1 to GPU│B2 to GPU│         │B3 to GPU│         │B4 to GPU│
                       └─────────┴─────────┘         └─────────┘         └─────────┘
                                 ┌───────────────────┬───────────────────┬───────────────────┐
GPU                              │INFER B1 into MODEL│INFER B2 into MODEL│INFER B3 into MODEL│
                                 └───────────────────┴───────────────────┴───────────────────┘
```


- Prefetching Implementation #1: `class data_prefetcher()` in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L265
- Prefetching Implementation #2: Sacrife 1 data loader process into a prefetcher process


- https://www.jpatrickpark.com/post/prefetcher/
- https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/

Achieving overlap between data transfers and other operations requires the use of CUDA streams, so first let’s learn about streams.




## Faster Model: TensorRT engine --into--> TorchScript module

- https://pytorch.org/TensorRT/
- https://pytorch.org/TensorRT/_notebooks/lenet-getting-started.html
- https://pytorch.org/TensorRT/py_api/ts.html?highlight=embed#torch_tensorrt.ts.embed_engine_in_new_module


## Summary

For getting fast training/inference 

- Data reading:
  - Use fast data staorage hardware (RAM, NVMe, RAID,...)
  - Use fast data decoding (libjpeg-turbo for images)
  - Even faster is you store precomted tensors and load them with either
    - Numpy.memmap
    - torch.Storage
    - FFIO
- Dataloader
  - Read images in parallel `num_workers`
  - Avoid unnecessary host copies `pin_memory=True`
  - Prefetching on CPU (CPU Queue) `prefetch_factor`
  - Prefetching on GPU (GPU Queue)
- Model
  - TensorRT





## CUDA Programming


Pytorch



```python

my_stream = torch.cuda.Stream()

with torch.cuda.stream(my_stream):
    
    # Send data to GPU (NO BLOCKING)
    data = data.cuda(non_blocking=True) # or data.to("cuda", non_blocking=True)
```


PyCUDA

```python

my_stream = cuda.Stream()

# Send data to GPU (NO BLOCKING)
cuda.memcpy_htod_async(dest=gpu_mem[name], src=cpu_mem[name], stream=my_stream)


cuda.memcpy_dtoh_async(dest=cpu_mem[name], src=gpu_mem[name], stream=my_stream)
```




# Copy betwwnn numpy and pytorch

|                  | Copy by value, Deep copy | Copy by reference, Shallow copy |
|------------------|--------------------------|---------------------------------|
| Numpy to Pytorch | `torch.tensor(my_npArr)` | `torch.from_numpy(my_npArr)`    |
| Pytorch to Numpy | `np.array(my_tensor)`    | `my_tensor.numpy()`             |






## Reference

- Paul Bridger [Twiter](https://twitter.com/paul_bridger), [Blog](paulbridger.com)
  - [Solving Machine Learning Performance Anti-Patterns: a Systematic Approach](https://paulbridger.com/posts/nsight-systems-systematic-optimization) Jun, 2021
  - [Object Detection at 2530 FPS with TensorRT and 8-Bit Quantization](https://paulbridger.com/posts/tensorrt-object-detection-quantized), Dec 2020
- Horace He [Twiter](https://twitter.com/cHHillee), [Blog](https://horace.io/writing.html)
  - [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) Mar, 2022
  - [Another thing PyTorch 2.0 helps speed up - overhead](https://twitter.com/cHHillee/status/1616906059368763392) Jan, 2023
- Jungkyu Park [Twiter](https://twitter.com/jpatrickpark) [Blog](https://www.jpatrickpark.com)
  - [Visualizing data loaders to diagnose deep learning pipelines](https://www.jpatrickpark.com/post/loader_sim) Apr, 2021
  - [Data Prefetching on GPU in Deep Learning](https://www.jpatrickpark.com/post/prefetcher) Feb, 2022 
- Yuxin Wu [Twiter](https://twitter.com/ppwwyyxx), [Blog](https://ppwwyyxx.com)
  - [Demystify RAM Usage in Multi-Process Data Loaders](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader)
- Christian S. Perone
  - https://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour
- [Building a Multi-Process Data Loader from Scratch](https://teddykoker.com/2020/12/dataloader/)



