Refactored from normalizing flow net. Code was getting too messy.
Implemented:
- normalizing flow
- continous normalizing flow
- conditional GAN
- conditional VAE

Installation:

```
pip install -r requirements.txt
```

Example usage:

```
python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_cnf --train_batch_size=16 --num_training_iters=50 --optimizer-name=adam --learning_rate=1e-5

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_gan --train_batch_size=16 --num_training_iters=50 --optimizer-name=adam --learning_rate=1e-5

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_gan_100d_noise --train_batch_size=20000 --num_training_iters=1000 --optimizer-name=adam --lr_g=5e-4 --lr_c=5e-4 --k=1 --n=1

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_cvae --train_batch_size=5000 --num_training_iters=100 --optimizer-name=adam --lr_g=1e-4

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_cvae_v3 --train_batch_size=128 --num_training_iters=1000 --optimizer-name=adam --lr_g=1e-3

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_cvae_v10 --train_batch_size=4096 --num_training_iters=2500 --optimizer-name=adam --lr_g=5e-5

python -m src.main --domain_name=sphere_100d --qd_config_name=112k --model_config_name=sphere_100d_cvae_v10 --train_batch_size=4096 --num_training_iters=5000 --optimizer-name=adam --lr_g=1e-4

python -m src.main --domain_name=arm_10d --qd_config_name=112k --model_config_name=arm_10d_cvae_v8 --train_batch_size=4096 --num_training_iters=5000 --optimizer-name=adam --lr_g=1e-4
```
