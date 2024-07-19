Refactored from normalizing flow net. Code was getting too messy.

Installation:

```
pip install -r requirements.txt
```

Distills archive for specified domains:

```
python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_nfn --train_batch_size=16 --num_training_iters=50 --optimizer=adam --learning_rate=1e-5
```