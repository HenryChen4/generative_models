import torch
import torch.nn as nn

from tqdm import trange, tqdm

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers,
                 activation,
                 device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        num_units = hidden_layers[0]
        depth = len(hidden_layers)

        self.layers = []
        # input layer
        self.layers.append(nn.Linear(self.input_dim,
                                         num_units,
                                         bias=False))
        self.layers.append(activation())
        
        # hidden layers
        for i in range(depth):
            self.layers.append(nn.Linear(num_units,
                                         num_units,
                                         bias=False))
            self.layers.append(nn.BatchNorm1d(num_units))
            self.layers.append(activation())
        
        #output layer
        self.layers.append(nn.Linear(num_units,
                                     output_dim,
                                     bias=False))
        
        self.model = nn.Sequential(*self.layers)

    def forward(self,
                x):
        return self.model(x)
        
class Encoder(nn.Module):
    def __init__(self, 
                 solution_dim,
                 latent_dim,
                 hidden_features,
                 activation,
                 device):
        super().__init__()
        self.solution_dim = solution_dim
        self.latent_dim = latent_dim
        self.device = device

        num_units = hidden_features[0]
        depth = len(hidden_features)

        self.layers = []
        # input layer
        self.layers.append(nn.Linear(solution_dim, 
                                     num_units,
                                     bias=False))
        self.layers.append(activation())
        # hidden layers
        for i in range(depth):
            self.layers.append(nn.Linear(num_units,
                                         num_units,
                                         bias=False))
            self.layers.append(nn.BatchNorm1d(num_units))
            self.layers.append(activation())
        # output layer
        self.layers.append(nn.Linear(num_units,
                                     latent_dim,
                                     bias=False))
        self.layers.append(activation())

        self.model = nn.Sequential(*self.layers)

    def forward(self, 
                solution):
        return self.model(solution)

class Decoder(nn.Module):
    def __init__(self,
                 solution_dim,
                 latent_dim,
                 hidden_features,
                 activation,
                 device):
        super().__init__()
        self.solution_dim = solution_dim
        self.latent_dim = latent_dim
        self.device = device

        num_units = hidden_features[0]    
        depth = len(hidden_features)

        self.layers = []
        # input layer
        self.layers.append(nn.Linear(latent_dim,
                                     num_units,
                                     bias=False))
        self.layers.append(activation())
        # hidden layers
        for i in range(depth):
            self.layers.append(nn.Linear(num_units,
                                         num_units,
                                         bias=False))
            self.layers.append(nn.BatchNorm1d(num_units))
            self.layers.append(activation())
        # output layer
        self.layers.append(nn.Linear(num_units,
                                     solution_dim,
                                     bias=False))

        self.model = nn.Sequential(*self.layers)
 
    def forward(self,
                solution):
        return self.model(solution)
    
class ConditionalVAE(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 context_mlp,
                 mu_mlp,
                 log_var_mlp):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context_mlp = context_mlp
        self.mu_mlp = mu_mlp
        self.log_var_mlp = log_var_mlp
    
    def condition(self,
                  latent,
                  context):
        return latent + self.context_mlp(context)

    def reparameterize(self, 
                       mu, 
                       log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self,
                solution,
                context):
        encoded = self.encoder(solution)

        mu = self.mu_mlp(encoded)
        log_var = self.log_var_mlp(encoded)

        latent = self.reparameterize(mu, log_var)
        # add the context
        conditioned_latent = self.condition(latent, context)
        decoded = self.decoder(conditioned_latent)

        return encoded, decoded, mu, log_var

def create_cvae(solution_dim,
                latent_dim,
                context_dim,
                encoding_config,
                decoding_config,
                context_config,
                mu_config,
                log_var_config,
                device):
    encoder = Encoder(solution_dim=solution_dim,
                      latent_dim=latent_dim,
                      hidden_features=encoding_config["hidden_layers"],
                      activation=encoding_config["activation"],
                      device=device).to(device)
    decoder = Decoder(solution_dim=solution_dim,
                      latent_dim=latent_dim,
                      hidden_features=decoding_config["hidden_layers"],
                      activation=decoding_config["activation"],
                      device=device)
    context_mlp = MLP(input_dim=context_dim,
                      output_dim=latent_dim,
                      hidden_layers=context_config["hidden_layers"],
                      activation=context_config["activation"],
                      device=device).to(device)
    mu_mlp = MLP(input_dim=latent_dim,
                 output_dim=latent_dim,
                 hidden_layers=mu_config["hidden_layers"],
                 activation=mu_config["activation"],
                 device=device).to(device)
    log_var_mlp = MLP(input_dim=latent_dim,
                      output_dim=latent_dim,
                      hidden_layers=log_var_config["hidden_layers"],
                      activation=log_var_config["activation"],
                      device=device).to(device)
    cvae = ConditionalVAE(encoder=encoder,
                          decoder=decoder,
                          context_mlp=context_mlp,
                          mu_mlp=mu_mlp,
                          log_var_mlp=log_var_mlp).to(device)

    return cvae

def train_cvae(cvae,
               train_loader,
               meas_obj_func,
               num_iters,
               optimizer,
               learning_rate,
               device):
    total_params = sum(p.numel() for p in cvae.parameters())
    print(f"Num params: {total_params}")    

    optimizer = optimizer(cvae.parameters(), lr=learning_rate)
    all_epoch_loss = []
    all_feature_error = []

    for epoch in trange(num_iters):
        epoch_loss = 0.
        feature_error = 0.
        batch_loss = 0.
        for i, (data_tuple) in tqdm(enumerate(train_loader)):
            solution_sample = data_tuple[0].to(device)
            context_sample = data_tuple[1].to(device)

            original_features = context_sample[:,:-1]

            _, decoded, mu, log_var = cvae(solution_sample,
                                           context_sample)
            
            _, features = meas_obj_func(decoded)
            batched_feature_err = torch.norm(features - original_features,
                                             p=2,
                                             dim=1)
            feature_error += batched_feature_err.mean()
            
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            criterion = nn.MSELoss()
            loss = criterion(decoded, solution_sample) + KLD
            batch_loss += loss

        batch_loss = batch_loss.mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        all_epoch_loss.append(batch_loss)
        all_feature_error.append(feature_error/len(train_loader))
        print(f"epoch: {epoch} \n"
              f"loss: {epoch_loss/len(train_loader)} \n"
              f"feature error: {feature_error/len(train_loader)}")
        
        del batch_loss
        
    return all_epoch_loss, all_feature_error

# === for testing only ===
class AutoEncoder(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, solution):
        encoded = self.encoder(solution)
        decoded = self.decoder(encoded)
        return encoded, decoded

def create_ae(solution_dim,
               latent_dim,
               hidden_features,
               activation,
               device):
    encoder = Encoder(solution_dim=solution_dim,
                      latent_dim=latent_dim,
                      hidden_features=hidden_features,
                      activation=activation,
                      device=device)
    decoder = Decoder(solution_dim=solution_dim,
                      latent_dim=latent_dim,
                      hidden_features=hidden_features,
                      activation=activation,
                      device=device)
    return AutoEncoder(encoder, decoder)

def train_ae(autoencoder,
             train_loader,
             num_iters,
             optimizer,
             learning_rate,
             device):
    optimizer = optimizer(autoencoder.parameters(), lr=learning_rate)
    all_epoch_loss = []
    for epoch in trange(num_iters):
        epoch_loss = 0.
        for _, (data_tuple) in tqdm(enumerate(train_loader)):
            encoded, decoded = autoencoder(data_tuple[0])
            criterion = nn.MSELoss()
            loss = criterion(decoded, data_tuple[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        all_epoch_loss.append(epoch_loss/len(train_loader))
        print(f"epoch: {epoch}, loss:{epoch_loss/len(train_loader)}")
    return all_epoch_loss