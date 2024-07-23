Refactored from normalizing flow net. Code was getting too messy.

Installation:

```
pip install -r requirements.txt
```

Example usage:

```
python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_cnf --train_batch_size=16 --num_training_iters=50 --optimizer-name=adam --learning_rate=1e-5

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_gan --train_batch_size=16 --num_training_iters=50 --optimizer-name=adam --learning_rate=1e-5

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_gan_100d_noise --train_batch_size=20000 --num_training_iters=1000 --optimizer-name=adam --lr_g=5e-4 --lr_c=5e-4 --k=1 --n=1
```