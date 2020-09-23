

################################# INPUT: 1 image    OUTPUT: 3 categorias

# https://towardsdatascience.com/advanced-dataloaders-with-fastai2-ecea62a7d97e
dblock = DataBlock(
  blocks=(ImageBlock(cls=PILImageBW), *(3*[CategoryBlock])),      # one image input and three categorical outputs
  getters=[ColReader('image_id', pref=train_path, suff='.png'),   # image input
           ColReader('grapheme_root'),                            # label 1
           ColReader('vowel_diacritic'),                          # label 2
           ColReader('consonant_diacritic')],                     # label 3
  n_inp=1,                                                        # Set the number of inputs
  splitter=IndexSplitter(df.loc[df.fold==fold].index),            # train/validation split
  batch_tfms=[Normalize.from_stats([0.0692], [0.2051]),           # Normalize the images with the specified mean and standard deviation
              *aug_transforms(do_flip=False, size=sz)])           # Add default transformations except for horizontal flip      

dls = dblock.dataloaders(df, bs=bs)                               # Create the dataloaders



################################# INPUT: 1 imagen    OUTPUT: 1 categoria multilabel

# https://medium.com/@kirankamath7/make-code-simple-with-datablock-api-part2-4064bd067bdc
myDB = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('fname', pref=str(path2/'train')+os.path.sep), # ColReader(0, pref=f'{planet_source}/train/', suff='.jpg'),
                   get_y=ColReader('labels', label_delim=' '), # ColReader(1, label_delim=' '),
                   item_tfms = [FlipItem(p=0.5),Resize(224,method='pad')],
                   batch_tfms=[*aug_transforms(do_flip=True,
                   								flip_vert=True,
                   								max_rotate=180.0,
                   								max_lighting=0.6,
                   								max_warp=0.1,
                   								p_affine=0.75,
                   								p_lighting=0.75,
                   								xtra_tfms=[RandomErasing(p=1.0, sh=0.1, min_aspect=0.2,max_count=2)]), Normalize])
dls = myDB.dataloaders(df)
dls.show_batch()
dls.show_batch(max_n=9, figsize=(12,9))


################################# INPUT: 2 textos     OUTPUT: 1 categoria

dblock = DataBlock(blocks=(
					    TextBlock.from_df(text_cols='text', res_col_name='text', rules=[]),
					    TextBlock.from_df(text_cols='text2', res_col_name='text2', trules=[]),
					    CategoryBlock
					), 
                    get_x=[ColReader('text'), ColReader('text2')], 
                    get_y=ColReader('label'), 
                    splitter=ColSplitter(col='is_valid'))





##############################3 CUSTOM BLOCK

class IdentityTransform(Transform):
    def __init__(self, prefix=None):
        self.prefix = prefix or ""
    def encodes(self, o):
        print(f"{self.prefix} encodes {o.__class__}")
        return o
    def decodes(self, o):
        print(f"{self.prefix} decodes {o.__class__}")
        return o




# DataBlock:
se debe partir de un dataframe maestro

- What is the types of your inputs/targets? (`Blocks`)
	- `RegressionBlock`: float targets
	- `CategoryBlock`: single-label categorical targets
	- `MultiCategoryBlock`: multi-label categorical targets
	- `TextBlock`: texts
	- `ImageBlock`: images
	- `MaskBlock`: segmentation masks (potentially with codes)
	- `PointBlock`: points in an image
	- `BBoxBlock`: bounding boxes in an image
	- `BBoxLblBlock`: labeled bounding boxes, potentially with vocab
- Where is your data? (`get_items`)
- Does something need to be applied to inputs? (`get_x`)
- Does something need to be applied to the target? (`get_y`)
- How to split the data? (`splitter`)
- Do we need to apply something on formed items? (`item_tfms`)
- Do we need to apply something on formed batches? (`batch_tfms`)



# Learner

[learn.show_training_loop()](https://dev.fast.ai/learner#Learner.show_training_loop)


Start Fit
   - begin_fit      : [TrainEvalCallback]
  Start Epoch Loop
     - begin_epoch    : []
    Start Train
       - begin_train    : [TrainEvalCallback]
      Start Batch Loop
         - begin_batch    : []
         - after_pred     : []
         - after_loss     : []
         - after_backward : []
         - after_step     : []
         - after_cancel_batch: []
         - after_batch    : [TrainEvalCallback]
      End Batch Loop
    End Train
     - after_cancel_train: []
     - after_train    : []
    Start Valid
       - begin_validate : [TrainEvalCallback]
      Start Batch Loop
         - **CBs same as train batch**: []
      End Batch Loop
    End Valid
     - after_cancel_validate: []
     - after_validate : []
  End Epoch Loop
   - after_cancel_epoch: []
   - after_epoch    : []
End Fit
 - after_cancel_fit: []
 - after_fit      : []




[learn.summary()](dasfadfasdf)

RegModel (Input shape: ['16 x 1'])
================================================================
Layer (type)         Output Shape         Param #    Trainable 
================================================================
RegModel             16 x 1               2          True      
________________________________________________________________

Total params: 2
Total trainable params: 2
Total non-trainable params: 0

Optimizer used: functools.partial(<function SGD at 0x7fd4eedaee60>, mom=0.9)
Loss function: FlattenedLoss of MSELoss()

Callbacks:
  - TrainEvalCallback
  - Recorder
  - ProgressCallback






def ImageBlock(cls=PILImage):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=IntToFloatTensor)


def CategoryBlock(vocab=None, add_na=False):
    "`TransformBlock` for single-label categorical targets"
    return TransformBlock(type_tfms=Categorize(vocab=vocab, add_na=add_na))


def RegressionBlock(n_out=None):
    "`TransformBlock` for float targets"
    return TransformBlock(type_tfms=RegressionSetup(c=n_out))

class RegressionSetup(Transform):
    "Transform that floatifies targets"
    def __init__(self, c=None): self.c = c
    def encodes(self, o): return tensor(o).float()
    def decodes(self, o): return TitledFloat(o) if o.ndim==0 else TitledTuple(o_.item() for o_ in o)
    def setups(self, dsets):
        if self.c is not None: return
        try: self.c = len(dsets[0]) if hasattr(dsets[0], '__len__') else 1
        except: self.c = 0

class Categorize(Transform):
    "Reversible transform of category string to `vocab` id"
    loss_func,order=CrossEntropyLossFlat(),1
    def __init__(self, vocab=None, add_na=False):
        self.add_na = add_na
        self.vocab = None if vocab is None else CategoryMap(vocab, add_na=add_na)

    def setups(self, dsets):
        if self.vocab is None and dsets is not None: self.vocab = CategoryMap(dsets, add_na=self.add_na)
        self.c = len(self.vocab)

    def encodes(self, o): return TensorCategory(self.vocab.o2i[o])
    def decodes(self, o): return Category      (self.vocab    [o])
