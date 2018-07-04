import javia.deeplearner


dl = deeplearner(model="resnet", data="my data folder")
dl.summary()
dl.lr_find()
dl.fit(epochs=2, lr=xxx, cycle_len=1, wd=1e-5)
dl.unfreeze()
dl.fit(epochs=2, lr=xxx, cycle_len=1)
