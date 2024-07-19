import torch
import numpy as np

from zuko.distributions import DiagNormal
from zuko.flows import Flow, GeneralCouplingTransform, UnconditionalDistribution, UnconditionalTransform
from zuko.transforms import MonotonicAffineTransform, PermutationTransform

from tqdm import trange, tqdm

def create_nfn(solution_dim,
               sigma,
               num_coupling_layers,
               num_context,
               hypernet_config,
               permute_seed): 
    """Creates an IKFlow normalizing flow network
    
    Args:
        solution_dim (int): Dimension of domain solution.
        sigma (float): Standard deviation of base solution distribution.
        num_coupling_layers (int): Number of coupling layers.
        num_context (int): Number of conditioning inputs.
        hypernet_config (dict): Dictionary for conditional neural network.
        permute_seed (permute_seed): Seed for coupling layer permutation.
    Returns:
        flow (Flow): Flow object (normalizing flow network).
    """
    transforms = []
    for i in range(num_coupling_layers):
        torch.manual_seed(permute_seed + i)

        single_transform = GeneralCouplingTransform(
            features=solution_dim,
            context=num_context,
            univariate=MonotonicAffineTransform,
            **hypernet_config
        )
        permute_transform = UnconditionalTransform(
            PermutationTransform,
            torch.randperm(solution_dim),
            buffer=True
        )

        transforms.append(single_transform)
        transforms.append(permute_transform)

    flow = Flow(
        transform=transforms,
        base=UnconditionalDistribution(
            DiagNormal,
            loc=torch.full((solution_dim, ), 0, dtype=torch.float32),
            scale=torch.full((solution_dim, ), sigma, dtype=torch.float32),
            buffer=True
        )
    )

    return flow

def distill_archive_nfn(nfn,
                        train_loader,
                        meas_obj_func,
                        num_iters,
                        optimizer,
                        learning_rate,
                        device):
    """Distills archive and trains normalizing flow net.

    Args:
        nfn (zuko.Flow): Normalizing flow archive model.
        train_loader (torch.DataLoader): Data loader with training data.
        meas_obj_func (def): Feature and objective function of solution.
        num_iters (int): Number of training iterations.
        optimizer (torch.nn.optim): Optimizer for neural net.
        learning_rate (float): Learning rate.
        device (str): cuda or cpu.
    Returns:
        all_epoch_loss (list): List of all loss accumulated.
        all_feature_dist (list): List of all feature error accumulated.
    """
    nfn.to(device)
    optimizer = optimizer(nfn.parameters(), lr=learning_rate)

    all_epoch_loss = []
    all_feature_dist = []

    for epoch in trange(num_iters):
        epoch_loss = 0.
        feature_dist = 0.

        for i, (data_tuple) in enumerate(tqdm(train_loader)):
            original_solution = data_tuple[0].to(device)
            original_context = data_tuple[1].to(device)
            original_features = original_context[:,:-1]

            # calculating loss
            batch_loss = -nfn(original_context).log_prob(original_solution)
            batch_loss = batch_loss.mean()
            epoch_loss += batch_loss.item()

            # calculating l2 norm between features
            generated_solutions = nfn(original_context).sample().to(device)
            generated_features, _ = meas_obj_func(generated_solutions.cpu().detach().numpy())
            all_feature_dist = torch.norm(generated_features - original_features,
                                          p=2,
                                          dim=1)
            mean_feature_dist = all_feature_dist.mean().to(device)
            feature_dist += mean_feature_dist

            # backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            clip_value = 0.5 # anything > is risky
            torch.nn.utils.clip_grad_value_(nfn.paramters(), clip_value)
            optimizer.step()
        
        print(f"epoch: {epoch}, loss: {epoch_loss/len(train_loader)},
              feature err: {feature_dist/len(train_loader)}")
        all_epoch_loss.append(epoch_loss/len(train_loader))
        all_feature_dist.append(feature_dist/len(train_loader))
    
    return all_epoch_loss, all_feature_dist
