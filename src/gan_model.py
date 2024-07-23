"""Conditional generative adversarial network"""

import torch
import torch.nn as nn

from tqdm import trange, tqdm

# TODO: Clean up code
# TODO: Normalize features to be between 0 and 1

class Generator(nn.Module):
    def __init__(self, 
                 solution_dim,
                 noise_dim,
                 num_context, 
                 hidden_features,
                 activation,
                 device):
        """ Network transforms input with noise dim + embedding 
        dim to output with solution dim.
        """

        super().__init__()
        self.solution_dim = solution_dim
        self.noise_dim = noise_dim
        self.device = device

        # generator model definition
        self.layers = []

        # input layer
        self.layers.append(nn.Linear(noise_dim + num_context,
                                     hidden_features[0],
                                     bias=False))
        self.layers.append(activation())

        # hidden layers
        for i in range(len(hidden_features)-1):
            self.layers.append(nn.Linear(hidden_features[0], 
                                         hidden_features[0],
                                         bias=False))
            self.layers.append(nn.BatchNorm1d(hidden_features[0]))
            self.layers.append(activation())

        # output layer
        self.layers.append(nn.Linear(hidden_features[0],
                                     solution_dim,
                                     bias=False))
        self.model = nn.Sequential(*self.layers)
    
    def forward(self,
                noise_sample, 
                context):
        # concat conditional info with overall input
        generator_input = torch.cat((noise_sample, context), dim=1).to(self.device)

        return self.model(generator_input)
    
class Critic(nn.Module):
    def __init__(self, 
                 solution_dim,
                 num_context, 
                 hidden_features,
                 activation,
                 device):
        """ Network takes a solution and its context and output probability
        of it being a real sample.
        """

        super().__init__()
        self.solution_dim = solution_dim
        self.device = device

        # generator model definition
        self.layers = []

        # input layer
        self.layers.append(nn.Linear(solution_dim + num_context,
                                     hidden_features[0],
                                     bias=False))
        self.layers.append(activation())

        # hidden layers
        for i in range(len(hidden_features)-1):
            self.layers.append(nn.Linear(hidden_features[0], 
                                         hidden_features[0],
                                         bias=False))
            self.layers.append(nn.BatchNorm1d(hidden_features[0]))
            self.layers.append(activation())

        # output layer
        self.layers.append(nn.Linear(hidden_features[0],
                                     1,
                                     bias=False))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)

    def forward(self, 
                solution_sample,
                context): 
        # concat conditional info with overall input
        generator_input = torch.cat((solution_sample, context), dim=1).to(self.device)

        return self.model(generator_input)

def create_gan(solution_dim,
               noise_dim,
               num_context,
               hidden_features,
               activation,
               device):
    generator = Generator(solution_dim=solution_dim,
                          noise_dim=noise_dim,
                          num_context=num_context,
                          hidden_features=hidden_features,
                          activation=activation,
                          device=device)
    critic = Critic(solution_dim=solution_dim,
                    num_context=num_context,
                    hidden_features=hidden_features,
                    activation=activation,
                    device=device)
    return generator, critic

def train_gan(generator,
              critic,
              train_loader,
              meas_obj_func,
              max_meas,
              num_iters,
              k,
              n,
              gen_optimizer,
              critic_optimizer,
              lr_g,
              lr_c,
              device):
    total_params = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"Num params: {total_params}")

    generator.to(device)
    critic.to(device)

    gen_optimizer = gen_optimizer(generator.parameters(), lr=lr_g)
    critic_optimizer = critic_optimizer(critic.parameters(), lr=lr_c)

    all_epoch_critic_loss = []
    all_epoch_gen_loss = []
    all_feature_err = []

    for epoch in trange(num_iters):
        mean_critic_loss = 0.
        mean_gen_loss = 0.
        mean_feature_err = 0.
        for i, (data_tuple) in enumerate(tqdm(train_loader)):
            batch_size = data_tuple[0].shape[0]

            true_label = torch.ones((batch_size, 1)).to(device)
            gen_label = torch.zeros((batch_size, 1)).to(device)

            z = torch.rand(size=(batch_size, generator.noise_dim)).to(device)

            # ===== update critic k amount of times =====
            for c in range(k): 
                loss_func = nn.BCELoss()
                # maximize log(d(x)) + log(1 - d(g(z))) --> you want to maximize ability to predict real data
                critic.zero_grad()
                # 1. sample true data 
                real_prob = critic.forward(solution_sample=data_tuple[0].to(device),
                                           context=data_tuple[1].to(device)/max_meas)
                real_loss = loss_func(real_prob, true_label)
                
                # 2. sample fake data 
                gen_data = generator.forward(noise_sample=z,
                                             context=data_tuple[1].to(device)/max_meas)
                gen_prob = critic.forward(solution_sample=gen_data,
                                          context=data_tuple[1].to(device)/max_meas)
                gen_loss = loss_func(gen_prob, gen_label)

                # 3. compute loss and backpropagate thru critic
                critic_loss = real_loss + gen_loss

                mean_critic_loss += critic_loss.item()

                critic_loss.backward()
                critic_optimizer.step()

            # ===== update generator n amount of time =====
            for c in range(n):
                loss_func = nn.BCELoss()
                # minimize log(1 - D(g(z))) --> you want to minimize the critic's ability to predict real data
                generator.zero_grad()
                # 1. sample fake data
                gen_data = generator.forward(noise_sample=z,
                                             context=data_tuple[1].to(device))
                
                # 1.5 recording feature error
                original_features = data_tuple[1][:,:-1]
                _, features = meas_obj_func(gen_data)
                batched_feature_err = torch.norm(features.to(device) - original_features.to(device),
                                                 p=2,
                                                 dim=1)
                mean_feature_err += batched_feature_err.mean().to(device)

                gen_prob = critic.forward(solution_sample=gen_data,
                                          context=data_tuple[1].to(device))
                gen_loss = loss_func(gen_prob, true_label)

                mean_gen_loss += gen_loss.item()

                # 2. backpropagate thru generator
                gen_loss.backward()
                gen_optimizer.step()
        
        print(f"epoch: {epoch} \n"
              f"critic loss: {mean_critic_loss/(k * len(train_loader))} \n"
              f"generator loss: {mean_gen_loss/(n * len(train_loader))} \n"
              f"feature error: {mean_feature_err/(n * len(train_loader))}")

        all_epoch_critic_loss.append(mean_critic_loss/(k * len(train_loader)))
        all_epoch_gen_loss.append(mean_gen_loss/(n * len(train_loader)))
        all_feature_err.append(mean_feature_err/(n * len(train_loader)))
    
    return all_epoch_critic_loss, all_epoch_gen_loss, all_feature_err

