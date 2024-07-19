import torch

from zuko.distributions import DiagNormal
from zuko.flows import CNF

from tqdm import trange, tqdm

def create_cnf(solution_dim,
               num_context,
               hypernet_config):
    """Creates a continous normalizing flow network
    
    Args:
    Returns:
    """
    flow = CNF(features=solution_dim,
               context=num_context,
               **hypernet_config)

    return flow

def distill_archive_cnf(cnf,
                        train_loader,
                        meas_obj_func,
                        num_iters,
                        optimizer,
                        learning_rate,
                        device):
    print(device)
    cnf.to(device)
    optimizer = optimizer(cnf.parameters(), lr=learning_rate)

    all_epoch_loss = []
    all_feature_err = []

    for epoch in trange(num_iters):
        epoch_loss = 0.
        feature_err = 0.

        for i, (data_tuple) in enumerate(tqdm(train_loader)):
            original_solution = data_tuple[0].to(device)
            original_context = data_tuple[1].to(device)
            original_features = original_context[:,:-1]

            # calculating loss
            batch_loss = -cnf(original_context).log_prob(original_solution)
            batch_loss = batch_loss.mean()
            epoch_loss += batch_loss.item()

            # calculating l2 norm between features
            generated_solution = cnf(original_context).sample().to(device)
            _, generated_features = meas_obj_func(generated_solution)
            all_feature_err = torch.norm(generated_features - original_features,
                                         p=2,
                                         dim=1)
            mean_feature_err = all_feature_err.mean().to(device)
            feature_err += mean_feature_err

            # backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            clip_value = 0.5
            torch.nn.utils.clip_grad_value_(cnf.parameters(), clip_value)
            optimizer.step()
        
        print(f"epoch: {epoch}, loss: {epoch_loss/len(train_loader)}, feature err: {feature_err/len(train_loader)}")
        all_epoch_loss.append(epoch_loss/len(train_loader))
        all_feature_err.append(feature_err/len(train_loader))
    
    return all_epoch_loss, all_feature_err