import argparse
import collections
import csv
import os

import torch
import yaml
from trainer import Trainer

exp_name = "airfield_fhl_ser"

def log_metrics(round_id, metrics, filename='nono/step.csv'):
    """
    Appends metrics to a CSV file. Writes header if file does not exist.
    
    Args:
        epoch (int): Current epoch number.
        metrics (tuple): (precision, recall, mAP@50, mAP)
        filename (str): Path to the CSV file.
    """
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'precision', 'recall', 'mAP@50', 'mAP'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'round': str(round_id + 1).zfill(3),
            'precision': f'{metrics[0]:.3f}',
            'recall': f'{metrics[1]:.3f}',
            'mAP@50': f'{metrics[2]:.3f}',
            'mAP': f'{metrics[3]:.3f}'
        })
        f.flush()

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

    data_paths = ['/home/mattias/Documents/projects/Wisard_usecase/datasets/dataset_Airfield',
                  '/home/mattias/Documents/projects/Wisard_usecase/datasets/dataset_FHL']
    

    with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
        params = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--local_updates", default=25, type=int)
    args = parser.parse_args()
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    trainers = {}
    for data_path in data_paths:
        name = data_path.split("/")[-1]
        trainers[name] = Trainer(args, params, data_path)

    global_model = trainers[name].model.state_dict()
    
    for round in range(1000):
        model_upd = []
        print("round: ", round)
        for trainer in trainers:
            trainers[trainer].model.load_state_dict(global_model, strict=True)
            m_pre, m_rec, map50, mean_ap = trainers[trainer].validate()
            #val_res = [m_pre, m_rec, map50, mean_ap]
            log_metrics(round, [m_pre, m_rec, map50, mean_ap], os.path.join(exp_name,trainer+'_step.csv'))

            trainers[trainer].train()
            model_upd += [trainers[trainer].model.state_dict()]
        global_model = aggregate(model_upd)


        
if __name__ == "__main__":
    main()