import javia.deeplearner


dl = deeplearner(model="resnet34", data="/home/javi/DL/fastai/courses/dl1/data/dogscats")
dl.change_last_layer(2)
dl.summary()
dl.lr_find()
dl.fit(epochs=2, lr=xxx, cycle_len=1, wd=1e-5)
dl.unfreeze()


dl.fit(epochs=2, batch_size=32, lr=xxx, cycle_len=1)


