# Python

### Pathlib

```python
from pathlib import Path

PATH = Path('data/pascal')
FILE = PATH / some_file
```

* PATH.iterdir()
* file object

### JSON

```python
import json

train_json = json.load(FILE.open()) # FILE is a path obj, and open() returns a file obj
train_json.keys()
```

### List and Dict comprehensions

```python
train_ids   = [o["id"]                 for o in train_json["images"]]      # list comprehension
train_files = {o["id"]: o["file_name"] for o in train_json["images"]}      # dict comprehension
categories  = {o["id"]: o["name"]      for o in train_json["categories"]}  # dict comprehension
```

### defaultdict

```python
import collections

train_anno = collections.defaultdict(lambda:[])

for o in train_j["annotations"]:
  if not o["ignore"]:
    bb o["bbox"]
    train_anno[o["id"]].append((bb, o["categories"]))
```


### Matplotlib
```python
def show_img(im, figsize=None, ax=None):
  if not ax: fig,ax = plt.subplots(figsize=figsize)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  return ax
```

TODO: https://medium.com/@pierre_guillou/fastai-how-to-start-663927d4db63
