import argparse
import collections
import os

import numpy as np
import torch
import yaml
from trainer import Trainer
from save_results import log_metrics
import copy

res_folder = 'results'
exp_name = "airfield_fhl_ser_lr-0.01-0.001-300"

def aggregate(models, copy_non_floats_from=0):
    """
    Aggregates model weights using FedAvg.
    
    Args:
        models: List of state_dicts from clients.
        global_model: The global model to be updated.
        copy_non_floats_from: Index of the client to copy non-float values from.
    """
    layers = models[0].keys()
    state_dict = collections.OrderedDict()

    for key in layers:
        tensors = [model[key] for model in models]
       
        if torch.is_floating_point(tensors[0]):
            # Average floating point tensors
            stacked = torch.stack(tensors, dim=0)
            state_dict[key] = torch.mean(stacked, dim=0)
        else:
            # Option 1: Skip entirely
            # continue

            # Option 2: Copy from a reference client
            state_dict[key] = models[copy_non_floats_from][key]

   
    return state_dict



def main():

    
    

    with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
        params = yaml.safe_load(f)
    params["lr0"] =0.01
    params["lrf"] =0.1

    dataset_path = params['dataset_path']
    data_paths = [os.path.join(dataset_path,'dataset_Airfield'),
                  os.path.join(dataset_path,'dataset_FHL')]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--local_updates", default=25, type=int)
    args = parser.parse_args()
    
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    if not os.path.exists(os.path.join(res_folder,exp_name)):
        os.makedirs(os.path.join(res_folder,exp_name))

    trainers = {}
    for data_path in data_paths:
        name = data_path.split("/")[-1]
        trainers[name] = Trainer(args, params, data_path)

    global_model = copy.deepcopy(trainers[name].model.state_dict())
    
    for round in range(2000):
        model_upd = []
        print("round: ", round)
        for trainer in trainers:
            trainers[trainer].model.load_state_dict(global_model, strict=True)

            m_pre, m_rec, map50, mean_ap = trainers[trainer].validate()
            learning_rate = trainers[trainer].optimizer.param_groups[0]["lr"]
            #val_res = [m_pre, m_rec, map50, mean_ap]
            log_metrics(round, [m_pre, m_rec, map50, mean_ap, learning_rate], os.path.join(res_folder,exp_name,trainer+'_step.csv'))

            trainers[trainer].train()

            model_upd += [trainers[trainer].model.state_dict()]
            print("lr: ", trainers[trainer].optimizer.param_groups[0]["lr"])

        global_model = copy.deepcopy(aggregate(model_upd))


        
if __name__ == "__main__":
    main()