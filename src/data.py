"""For gathering training samples from archive"""
import sys
import numpy as np
from tqdm import trange

import torch
from torch.utils.data import Dataset, DataLoader

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

class Archive_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def create_loader(data, batch_size, shuffle=True):
    x = [item[0] for item in data]
    y = [item[1] for item in data]

    dataset = Archive_Dataset(x, y)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)

    return dataloader

def create_scheduler(solution_dim,
                     solution_bounds,
                     measure_ranges,
                     init_sol,
                     grid_cells,
                     sigma0,
                     batch_size,
                     num_emitters):
    """Creates archive and scheduler to store training samples.
    
    Args:
        solution_dim (int): Dimension of solution for specified domain.
        solution_bounds (list): Solution bounds for sampling new solutions.
        measure_ranges (list): Measure bounds.
        init_sol (list): Initial parent solution.
        grid_cells (tuple): Number of cells in archive.
        sigma0 (float): Initial step size.
        batch_size (int): Number of children to sample.
        num_emitters (int): Number of emitters used
    Returns:
        archive (ribs.archive.GridArchive): Archive to store all solutions in.
        scheduler (ribs.Scheduler): Scheduler that inserts solutions.
    """
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=grid_cells,
        ranges=measure_ranges,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=init_sol,
            sigma0=sigma0,
            ranker="2imp",
            bounds=solution_bounds,
            batch_size=batch_size,
        ) for _ in range(num_emitters)
    ]

    scheduler = Scheduler(archive, emitters)

    return archive, scheduler

def fill_archive(obj_meas_func, 
                 scheduler, 
                 archive, 
                 num_iters):
    """Runs QD algorithm and returns archive with elites and
    all solutions leading to archive creation.
    
    Args:
        obj_meas_func (def): Feature and objective function.
        scheduler (ribs.Scheduler): Scheduler to fill archive.
        archive (ribs.archive.GridArchive): Archive to store solutions.
        num_iters (int): QD run iteration count.
    Returns:
        archive (ribs.archive.GridArchive): Elite filled archive.
        en_route_sols (list): All solutions generated ever.
    """
    en_route_sols = [] # store all solutions generated ever
    for itr in trange(1, num_iters + 1, desc='Iterations', file=sys.stdout):
        sols = scheduler.ask()
        sols = torch.tensor(sols)
        objs, meas = obj_meas_func(sols)

        if len(sols) != len(objs) or len(sols) != len(meas):
            raise("Size mismatch, ribs issue.")
        
        for i in range(len(sols)):
            solution_dict = {
                "solution": None,
                "objective": None,
                "measures": None
            }

            solution_dict["solution"] = sols[i]
            solution_dict["objective"] = objs[i]
            solution_dict["measures"] = meas[i]

            en_route_sols.append(solution_dict)

        scheduler.tell(objs, meas)

    return archive, en_route_sols

# example: gather_solutions(qd_config, train_batch_size, **domain)
def gather_solutions(qd_config,
                     train_batch_size,
                     domain_name,
                     obj_meas_func,
                     solution_dim,
                     solution_bounds,
                     initial_sol,
                     feature_low,
                     feature_high):
    """Returns train loader with solutions generated by QD algo."""
    
    print(f"> Gathering solutions for {domain_name} domain")
    all_sols = []

    archive, scheduler = create_scheduler(solution_dim=solution_dim,
                                          solution_bounds=solution_bounds,
                                          measure_ranges=[[feature_low[0], feature_high[0]],
                                                          [feature_low[1], feature_high[1]]],
                                          init_sol=initial_sol,
                                          grid_cells=qd_config["grid_cells"],
                                          sigma0=qd_config["sigma0"],
                                          batch_size=qd_config["batch_size"],
                                          num_emitters=qd_config["num_emitters"])

    archive, en_route_sols = fill_archive(obj_meas_func=obj_meas_func,
                                          scheduler=scheduler,
                                          archive=archive,
                                          num_iters=qd_config["num_qd_iters"])
    
    for sol in en_route_sols:
        arm_pose = sol["solution"]
        objective = sol["objective"]
        measures = sol["measures"]

        single_train_tuple = (
            torch.tensor(arm_pose.detach().numpy(), dtype=torch.float32),
            torch.cat((torch.tensor(measures.detach().numpy(), dtype=torch.float32), 
                       torch.tensor(objective.detach().numpy(), dtype=torch.float32).unsqueeze(dim=0)))
        )

        all_sols.append(single_train_tuple)

    for elite in archive:
        arm_pose = elite["solution"]
        objective = elite["objective"]
        measures = elite["measures"]

        single_train_tuple = (
            torch.tensor(arm_pose, dtype=torch.float32),
            torch.cat((torch.tensor(measures, dtype=torch.float32), 
                       torch.tensor(objective, dtype=torch.float32).unsqueeze(dim=0)))
        )
        
        all_sols.append(single_train_tuple)

    data_loader = create_loader(all_sols, train_batch_size, shuffle=True)
    return data_loader