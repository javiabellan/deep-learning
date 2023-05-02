
## Parameter Efficient FineTunning (PEFT) techniques

PEFT approaches only **fine-tune a small number of (extra) model parameters** while freezing most parameters of the pretrained LLMs, thereby greatly decreasing the computational and storage costs. This also overcomes the issues of [catastrophic forgetting](https://arxiv.org/abs/1312.6211), a behaviour observed during the full finetuning of LLMs. PEFT approaches have also shown to be better than fine-tuning in the low-data regimes and generalize better to out-of-domain scenarios. It can be applied to various modalities, e.g., image classification and stable diffusion dreambooth.

It also helps in portability wherein users can tune models using PEFT methods to get tiny checkpoints worth a few MBs compared to the large checkpoints of full fine-tuning

### Advantages
- Faster/Cheaper finetuning (with smaller GPUs)
- Avoids catastrophic forgetting
- Good fine-tuning in the low-data regimes 
- Portability (tiny checkpoints)


### PEFT methods:

- **LoRA** [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) Jun 2021
  - [LoRA Explained in Detail - Fine-Tune your LLM on your local GPU](https://www.youtube.com/watch?v=YVU5wAA6Txo)
  - [Boost Fine-Tuning Performance of LLM: Optimal Architecture w/ PEFT LoRA Adapter-Tuning on Your GPU](https://www.youtube.com/watch?v=A-a-l_sFtYM)
- Prefix Tuning: P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
- Prompt Tuning: The Power of Scale for Parameter-Efficient Prompt Tuning
- P-Tuning: GPT Understands, Too
- **Adapter** [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/abs/2303.16199) Mar 2023

### PEFT reference
- https://huggingface.co/blog/peft
- https://github.com/huggingface/peft
- [Fine-tuning de grandes modelos de lenguaje con Manuel Romero](https://www.youtube.com/watch?v=WYcJb8gYBZU)