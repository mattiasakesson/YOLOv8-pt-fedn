import numpy as np
import torch
from dataloader import get_concatenated_dataloader
from trainer import Trainer
import os
import yaml
import argparse
import csv
from trainer import PersistentDataLoader

res_folder = 'results'
os.makedirs(res_folder, exist_ok=True)


exp_name = 'lean_ml_mterie'


exp_name = os.path.join(res_folder, exp_name)
weights_dir = os.path.join(exp_name,"modelstates")
#os.makedirs(weights_dir, exist_ok=True)

submapp = os.path.join(exp_name,"sub_mapp")
validation_folder = os.path.join(res_folder, "Other_res")
submapp = validation_folder
os.makedirs(submapp, exist_ok=True)


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

def main():

    with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
        params = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--local_updates", default=70, type=int)
    args = parser.parse_args()
    
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    dataset_path = params['dataset_path']
    client_names_ = [ds for ds in os.listdir(dataset_path) if ds != 'dataset_Others' and ds.startswith("dataset") and not ds.endswith("old") and not ds.endswith("Other")]
    
    client_names = [os.path.join(dataset_path,ds) for ds in client_names_ if ds]
    for ds in client_names_:
        print(ds)
    client_names = [os.path.join(dataset_path,"dataset_Other")]
    #quit()
    

    
   
    val_clients = {}
    
    for client_name in client_names:
        val_clients[client_name.split("/")[-1]] = Trainer(args, params, data_path=client_name, trainer_mode=False)
    
    epoch = 1124

    state_dict = torch.load(os.path.join(weights_dir,"modelstate_"+str(epoch)))
    cl_names = list(val_clients.keys())
    model = val_clients[cl_names[0]].model
    model.load_state_dict(state_dict)
    #tresholds = [0.001, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    tresholds = np.arange(0.5,0.7,0.01)
    tresholds = [0.1]
    for treshold in tresholds:
        
        mean_pre, mean_rec, mean_map50, mean_map = [], [], [], []
        # ema model evaluation
        for val_client in val_clients:
            print("val_client: ", val_client)
            name = val_client
            #val_clients[val_client].model.load_state_dict(trainer.model.state_dict())
            m_pre, m_rec, map50, mean_ap = val_clients[val_client].validate(model,treshold)
            mean_pre += [m_pre]
            mean_rec += [m_rec]
            mean_map50 += [map50]
            mean_map += [mean_ap]
            #log_metrics(epoch, [m_pre, m_rec, map50, mean_ap], os.path.join(submapp,name+'_ema_step_tr_'+str(treshold)+'.csv'))
            name= exp_name.split("/")[-1]
            print("name: ", name)
            print(os.path.join(submapp,name+''+str(treshold)+'.csv'))

            log_metrics(epoch, [m_pre, m_rec, map50, mean_ap], os.path.join(submapp,name+''+str(treshold)+'.csv'))

        print("mean pre: ", np.mean(mean_pre))
        print("mean recall: ", np.mean(mean_rec))
        print("mean map50: ", np.mean(mean_map50))
        print("mean recall: ", np.mean(mean_map))


            
            
            



if __name__ == "__main__":
    main()