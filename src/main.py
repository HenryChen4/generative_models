import os
import fire
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ranger_adabelief import RangerAdaBelief as Ranger

from src.data import gather_solutions
from src.domains import DOMAIN_CONFIGS, arm, sphere
from src.visualize import visualize

from src.nfn_model import create_nfn, distill_archive_nfn
from src.cnf_model import create_cnf, distill_archive_cnf
from src.gan_model import create_gan, train_gan
from src.cvae_model import create_cvae, train_cvae

def get_qd_config(config_name):
    # names correspond to roughly number of samples
    qd_configs = {
        "112k": {
            "grid_cells": (100, 100),
            "sigma0": 0.1,
            "batch_size": 30,
            "num_emitters": 5,
            "num_qd_iters": 700
        }
    }
    return qd_configs[config_name]

def get_model_config(config_name):
    models = {
        "arm_10d_nfn": {
            "solution_dim": 10,
            "sigma": torch.pi/3,
            "num_coupling_layers": 15,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (1024, 1024, 1024),
                "activation": nn.LeakyReLU
            },
            "permute_seed": 41534
        },
        "arm_10d_cnf": {
            "solution_dim": 10,
            "sigma": None,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (1024, 1024, 1024),
                "activation": nn.ELU
            },
            "type": "cnf"
        },
        "arm_10d_cnf_ffj": {
            "solution_dim": 10,
            "sigma": torch.pi/3,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (1024, 1024, 1024),
                "activation": nn.ELU
            },
            "type": "ffj"
        },
        "arm_100d_cnf_ffj": {
            "solution_dim": 100,
            "sigma": torch.pi/3,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (1024, 1024, 1024),
                "activation": nn.ELU
            },
            "type": "ffj"
        },
        "arm_10d_gan_100d_noise": {
            "solution_dim": 10,
            "noise_dim": 100,
            "num_context": 3,
            "hidden_features": (1024, 1024, 1024),
            "activation": nn.ReLU,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "arm_10d_gan_150d_noise": {
            "solution_dim": 10,
            "noise_dim": 150,
            "num_context": 3,
            "hidden_features": (512, 512, 512),
            "activation": nn.ReLU,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "arm_10d_cvae_v1": {
            "solution_dim": 10,
            "latent_dim": 8,
            "context_dim": 3,
            "encoding_config": {
                "hidden_layers": [512, 512, 512],
                "activation": nn.ReLU,
            },
            "decoding_config": {
                "hidden_layers": [512, 512, 512],
                "activation": nn.ReLU,
            },
            "context_config": {
                "hidden_layers": [32, 32],
                "activation": nn.ReLU,
            },
            "mu_config": {
                "hidden_layers": [64, 64],
                "activation": nn.ReLU,
            },
            "log_var_config": {
                "hidden_layers": [64, 64],
                "activation": nn.ReLU,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "arm_10d_cvae_v4": {
            "solution_dim": 10,
            "latent_dim": 50,
            "context_dim": 3,
            "encoding_config": {
                "hidden_layers": [2048, 2048, 2048],
                "activation": nn.LeakyReLU,
            },
            "decoding_config": {
                "hidden_layers": [2048, 2048, 2048],
                "activation": nn.LeakyReLU,
            },
            "context_config": {
                "hidden_layers": [128, 128],
                "activation": nn.ReLU,
            },
            "mu_config": {
                "hidden_layers": [128, 128, 128],
                "activation": nn.ReLU,
            },
            "log_var_config": {
                "hidden_layers": [128, 128, 128],
                "activation": nn.ReLU,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "arm_10d_cvae_v5": {
            "solution_dim": 10,
            "latent_dim": 200,
            "context_dim": 3,
            "encoding_config": {
                "hidden_layers": [1024, 1024, 1024, 1024],
                "activation": nn.ReLU,
            },
            "decoding_config": {
                "hidden_layers": [1024, 1024, 1024, 1024],
                "activation": nn.ReLU,
            },
            "context_config": {
                "hidden_layers": [64, 64],
                "activation": nn.ReLU,
            },
            "mu_config": {
                "hidden_layers": [64, 64, 64],
                "activation": nn.ReLU,
            },
            "log_var_config": {
                "hidden_layers": [64, 64, 64],
                "activation": nn.ReLU,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
    return models[config_name]

def main(domain_name,
         qd_config_name,
         model_config_name,
         train_batch_size,
         num_training_iters,
         optimizer_name,
         lr_g=5e-4,
         lr_c=5e-4,
         k=1,
         n=1):
    # 1. gather solutions from archive
    print("> Gathering training samples")
    domain_config = DOMAIN_CONFIGS[domain_name]
    qd_config = get_qd_config(qd_config_name)

    train_loader = gather_solutions(qd_config=qd_config,
                                    train_batch_size=train_batch_size,
                                    **domain_config)
    print(f"> Successfully gathered ~{len(train_loader) * train_batch_size} training samples!")

    # 2. create generative model
    model_config = get_model_config(model_config_name)
    archive_model = None
    critic = None
    if "nfn" in model_config_name:
        archive_model = create_nfn(**model_config)
    elif "cnf" in model_config_name:
        archive_model = create_cnf(**model_config)
    elif "gan" in model_config_name:
        archive_model, critic = create_gan(**model_config)
    elif "cvae" in model_config_name:
        archive_model = create_cvae(**model_config)

    # 3. train archive model and save results
    archive_model_optimizer = None
    critic_optimizer = None
    if optimizer_name == "adam":
        archive_model_optimizer = torch.optim.Adam
        critic_optimizer = torch.optim.Adam
    elif optimizer_name == "ranger":
        archive_model_optimizer = Ranger
        critic_optimizer = Ranger
    else:
        print(f"{optimizer_name} is not yet implemented!")

    if "nfn" in model_config_name:
        all_epoch_loss, all_feature_err = distill_archive_nfn(nfn=archive_model,
                                                              train_loader=train_loader,
                                                              meas_obj_func=domain_config["obj_meas_func"],
                                                              num_iters=num_training_iters,
                                                              optimizer=archive_model_optimizer,
                                                              learning_rate=lr_g,
                                                              device="cuda" if torch.cuda.is_available() else "cpu")
    elif "cnf" in model_config_name:
        all_epoch_loss, all_feature_err = distill_archive_cnf(cnf=archive_model,
                                                              train_loader=train_loader,
                                                              meas_obj_func=domain_config["obj_meas_func"],
                                                              num_iters=num_training_iters,
                                                              optimizer=archive_model_optimizer,
                                                              learning_rate=lr_g,
                                                              device="cuda" if torch.cuda.is_available() else "cpu")
    elif "gan" in model_config_name:
        all_epoch_c_loss, all_epoch_g_loss, all_feature_err = train_gan(generator=archive_model,
                                                                        critic=critic,
                                                                        train_loader=train_loader,
                                                                        meas_obj_func=domain_config["obj_meas_func"],
                                                                        max_meas=domain_config["feature_high"][0],
                                                                        num_iters=num_training_iters,
                                                                        k=k,
                                                                        n=n,
                                                                        gen_optimizer=archive_model_optimizer,
                                                                        critic_optimizer=critic_optimizer,
                                                                        lr_g=lr_g,
                                                                        lr_c=lr_c,
                                                                        device="cuda" if torch.cuda.is_available() else "cpu")
    elif "cvae" in model_config_name:
        all_epoch_loss, all_feature_err = train_cvae(cvae=archive_model,
                                                     train_loader=train_loader,
                                                     meas_obj_func=domain_config["obj_meas_func"],
                                                     num_iters=num_training_iters,
                                                     optimizer=archive_model_optimizer,
                                                     learning_rate=lr_g,
                                                     device="cuda" if torch.cuda.is_available() else "cpu")
        
        cpu_feature_err = []
        for err in all_feature_err:
            cpu_feature_err.append(err.cpu().detach().numpy())
        
        save_dir = f"results/archive_distill/{domain_name}/{model_config_name}"
        os.makedirs(save_dir, exist_ok=True)
        loss_save_path = os.path.join(save_dir, f"loss.png")
        error_save_path = os.path.join(save_dir, f"feature_error.png")
        model_save_path = os.path.join(save_dir, f"model.pth")

        torch.save(archive_model, model_save_path)

        plt.plot(np.arange(num_training_iters), all_epoch_loss, color="blue", label="loss")
        plt.savefig(loss_save_path)
        plt.clf()

        plt.plot(np.arange(num_training_iters), cpu_feature_err, color="blue", label="feature error")
        plt.savefig(error_save_path)
        plt.clf()

    # elif "cvae" in model_config_name:
    #     all_epoch_loss = dummy_train(autoencoder=archive_model,
    #                                  train_loader=train_loader,
    #                                  num_iters=num_training_iters,
    #                                  optimizer=archive_model_optimizer,
    #                                  learning_rate=lr_g,
    #                                  device=None)
    #     # visualizing original arms
    #     og_arms = next(iter(train_loader))[0]

    #     print(og_arms)

    #     _, context = domain_config["obj_meas_func"](og_arms[:10])
    #     link_lengths = np.ones(shape=(len(og_arms), ))
    #     objectives = np.ones(shape=(len(og_arms), ))
    #     _, ax = plt.subplots()
    #     visualize(solutions=og_arms[:10],
    #               link_lengths=link_lengths,
    #               objectives=objectives,
    #               ax=ax,
    #               context=context)
    #     plt.show()

    #     # visualizing encoded and decoded arms
    #     _, decoded_arms = archive_model(og_arms)

    #     print(decoded_arms)

    #     _, context = domain_config["obj_meas_func"](decoded_arms[:10])
    #     link_lengths = np.ones(shape=(len(decoded_arms.detach().numpy()), ))
    #     objectives = np.ones(shape=(len(decoded_arms.detach().numpy()), ))
    #     _, ax = plt.subplots()
    #     visualize(solutions=decoded_arms.detach().numpy()[:10],
    #               link_lengths=link_lengths,
    #               objectives=objectives,
    #               ax=ax,
    #               context=context.detach().numpy())
    #     plt.show()

    else:
        print(f"Check {model_config_name} exists")

if __name__ == "__main__":
    fire.Fire(main)

# TODO: FINISH VISUALIZING ARMS B4 and AFTER ENCODING DECODING ** DONE
# TODO: WRITE CONDITIONAL VAE ** DONE
# TODO: EVALUATE PERFORMANCE WITH HEATMAP AND MORE MEASUREMENTS