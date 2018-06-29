# Hardware

### GPU
Plenty of RAM and good single precision. Choose Nvidia.
 * GTX 1080ti (12GB RAM) is the price performance sweet spot today
 * GTX 1070 (8GB RAM) is also fine.
 * Dual GPU let you continue to prototype (jupyter) whilst running an experiment (training).
 
### CPU
 * i5 or i7 CPU is fine
 * **Number of cores**: The rule of thumb is to have two or more threads (or cores) per GPU in your system.
 * **Speed**: Important for computer vision and data augmentation.
 * **Other purposes**: Good CPUs are important for Gaming and VR/AR.
 * Full 40 PCIe lanes and correct PCIe spec (same as your motherboard) ???

### RAM
* As much RAM as possible.
* Note that you should really have more RAM than your total GPUs RAM

### Motherboard
* Make sure it supports >=64GB of RAM.
* Ensure you have x8 PCI lanes for each of your GPUs.
* For multiple GPUs:
  * Several PCIe 3.0 slots (I think it should be x16e).
  * Nvidia SLI support
 
### Storage
Fast drives (SSD or NVMe) for analysis, and big disks (HDD) for data
 * NVMe drivers are amazing! Get the biggest you can afford
 * Fill up the rest of space with the largest standard hard-drives you can buy.

### Power supply
* Make sure your power supply is more than sufficient.
* Rule of thumb: GPUs + CPU + (100..300).
* GTX 1080 requires 180 W of power and recommended PSU is 500 W or greater.
* Since I plan on adding second card later (and just in case), I opted for 850 W power supply.

> ## Laptop
> If you are considering buying a laptop. I recommend you:
> * GPU: GTX1060 6GB or better
> * Storage: With SSD or NVMe

## Examples

### Tight budget

| Component       | Option           | Price  |
| --------------- | ---------------- | ------ |
| GPU             | GTX 1070         | ???€   |
| CPU             | i5 or AMD fx6300 | ???€   |
| RAM             | 8GB or 16GB      | ???€   |
| Fast Storage    | SDD 128GB        | ???€   |
| Data Storage    | HDD 1T           | ???€   |
| Motherboard     | ???              | ???€   |
| Power supply    | ???W             | ???€   |

### Medium budget

| Component       | Option           | Price  |
| --------------- | ---------------- | ------ |
| GPU             | GTX 1080ti       | ???€   |
| CPU             | i5 or i7         | ???€   |
| RAM             | 16GB or 32GB     | ???€   |
| Fast Storage    | SDD or NVMe      | ???€   |
| Data Storage    | HDD              | ???€   |
| Motherboard     | ???              | ???€   |
| Power supply    | ???W             | ???€   |


### High budget

| Component       | Option           | Price  |
| --------------- | ---------------- | ------ |
| GPU             | 2x GTX 1080ti    | ???€   |
| CPU             | i7               | ???€   |
| RAM             | 32GB or 64GB     | ???€   |
| Fast Storage    | NVMe             | ???€   |
| Data Storage    | HDD              | ???€   |
| Motherboard     | ???              | ???€   |
| Power supply    | ???W             | ???€   |
