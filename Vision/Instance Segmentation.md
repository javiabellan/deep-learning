## Instance Segmentation

Instance segmentation is a challenging task, as it requires instance-level and pixel-wise predictions simultaneously.

3 categories: top-down, bottom-up, and single-stage methods.

1. Top-down methods: detect-then-segment paradigm.
- Mask R-CNN family follow the 

Bottom-up methods: label-then-cluster problem. (learn per-pixel embeddings and then cluster them into instance groups)

Singlestage instance segmentation framework on the top of onestage detectors.
- YOLACT (Bolya et al. 2019)
- CondInst (Tian, Shen, and Chen 2020)
- SOLO (Wang et al. 2020b) 

Queries instance segmentation frameworks (DETR style) eliminating NMS postprocessing.
- QueryInst (Fang et al. 2021)
- SOLQ (Dong et al. 2021)

However, they still need RoI cropping to separate different instances first, which may have the same limitations of the detect-then-segment pipeline.


In this paper, we
go for an end-to-end instance segmentation framework that
neither relies on RoI cropping nor NMS post-processing