from pytorch_lightning.utilities.cli import LightningCLI

cli = LightningCLI()



#  fit         Runs the full optimization routine.
#  validate    Perform one evaluation epoch over the validation set.
#  test        Perform one evaluation epoch over the test set.
#  predict     Run inference on your data.
#  tune        Runs routines to tune hyperparameters before training.


$ python trainer.py test --trainer.limit_test_batches=10 [...]




####### CONFIG FILES

trainer:
  max_epochs: 10
  limit_train_batches: 100
  callbacks:
    - class_path: pytorch_lightning.callbacks.EarlyStopping
      init_args:
        patience: 5
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
      init_args:
        ...

# python trainer.py fit --print_config
# python trainer.py fit --config experiment_defaults.yaml --trainer.max_epochs 100

# python trainer.py --config config1.yaml \
                    --config config2.yaml test --config config3.yaml [...]

######## TRAIN 

# python trainer.py fit \
#   --optimizer=Adam \
#   --optimizer.lr=0.01 \
#   --lr_scheduler=ExponentialLR \
#   --lr_scheduler.gamma=0.1 \
#   --trainer.callbacks=EarlyStopping \
#   --trainer.callbacks.patience=5 \
#   --trainer.callbacks=LearningRateMonitor \
#   --trainer.callbacks.logging_interval=epoch
#   --trainer.accelerator: null
#   --trainer.accumulate_grad_batches: 1
#   --trainer.amp_backend: native
#   --trainer.amp_level: O2
    --model=MyModel
    --model.feat_dim=64
    --data=MyData

# python trainer.py test chpt_path="ypur_best_model.ckpt"
# python trainer.py test --trainer.limit_test_batches=10