import os
import fire
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ranger_adabelief import RangerAdaBelief as Ranger

from src.data import gather_solutions
from src.nfn_model import create_nfn, distill_archive_nfn
from src.cnf_model import create_cnf, distill_archive_cnf
from src.domains import DOMAIN_CONFIGS, arm, sphere

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
        }
    }
    return models[config_name]

def main(domain_name,
         qd_config_name,
         model_config_name,
         train_batch_size,
         num_training_iters,
         optimizer_name,
         learning_rate):
    # 1. gather solutions from archive
    print("> Gathering training samples")
    domain_config = DOMAIN_CONFIGS[domain_name]
    qd_config = get_qd_config(qd_config_name)

    train_loader = gather_solutions(qd_config=qd_config,
                                    train_batch_size=train_batch_size,
                                    **domain_config)
    print(f"> Successfully gathered ~{len(train_loader) * train_batch_size} training samples!")

    # 2. create flow model
    algo_config = get_model_config(model_config_name)
    archive_model = None
    if "nfn" in model_config_name:
        archive_model = create_nfn(**algo_config)
    elif "cnf" in model_config_name:
        archive_model = create_cnf(**algo_config)
    
    # 3. train flow model
    optimizer = None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam
    elif optimizer_name == "ranger":
        optimizer = Ranger
    else:
        print(f"{optimizer_name} is not yet implemented!")

    all_epoch_loss = []
    all_feature_err = []    
    if "nfn" in model_config_name:
        all_epoch_loss, all_feature_err = distill_archive_nfn(nfn=archive_model,
                                                              train_loader=train_loader,
                                                              meas_obj_func=domain_config["obj_meas_func"],
                                                              num_iters=num_training_iters,
                                                              optimizer=optimizer,
                                                              learning_rate=learning_rate,
                                                              device="cuda" if torch.cuda.is_available() else "cpu")
    elif "cnf" in model_config_name:
        all_epoch_loss, all_feature_err = distill_archive_cnf(cnf=archive_model,
                                                              train_loader=train_loader,
                                                              meas_obj_func=domain_config["obj_meas_func"],
                                                              num_iters=num_training_iters,
                                                              optimizer=optimizer,
                                                              learning_rate=learning_rate,
                                                              device="cuda" if torch.cuda.is_available() else "cpu")

    # 4. save results
    cpu_epoch_loss = []
    cpu_mean_dist = []

    for i in all_epoch_loss:
        cpu_epoch_loss.append(i)

    for i in all_feature_err:
        cpu_mean_dist.append(i.cpu().numpy())

    # save results and model
    save_dir = f"results/archive_distill/{domain_name}/{model_config_name}"
    os.makedirs(save_dir, exist_ok=True)
    loss_and_dist_save_path = os.path.join(save_dir, f'loss_and_err.png')
    model_save_path = os.path.join(save_dir, f'model.pth')

    torch.save(archive_model, model_save_path)

    plt.plot(np.arange(num_training_iters), cpu_epoch_loss, color="green", label="log loss")
    plt.plot(np.arange(num_training_iters), cpu_mean_dist, color="blue", label="feature error")
    plt.legend()
    plt.savefig(loss_and_dist_save_path)
    plt.clf()

if __name__ == "__main__":
    fire.Fire(main)