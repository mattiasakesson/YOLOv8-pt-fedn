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


exp_name = 'lean_ml_carnation'
exp_name = os.path.join(res_folder, exp_name)
weights_dir = os.path.join(exp_name,"modelstates")
weights_dir_ema = os.path.join(exp_name,"modelstates_ema")

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(weights_dir_ema, exist_ok=True)



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
    params["lr0"] =0.001
    params["lrf"] =1.0
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    #client_names = ['/home/niklas/fedn-ultralytics-tutorial/datasets/dataset_FHL',
    #'/home/niklas/fedn-ultralytics-tutorial/datasets/dataset_Airfield']
    dataset_path = params['dataset_path']
    client_names_ = [ds for ds in os.listdir(dataset_path) if ds != 'dataset_Others' and ds.startswith("dataset") and not ds.endswith("old") and not ds.endswith("Other")]
    
    client_names = [os.path.join(dataset_path,ds) for ds in client_names_ ]
    for ds in client_names_:
        print(ds)
    

    client_names = [os.path.join(dataset_path,'dataset_Carnation')]
    name = 'Carnation'
    #                os.path.join(dataset_path,'dataset_FHLnew'),
    #                os.path.join(dataset_path,'dataset_Carnationnew')]
    #print(client_names)
    #train_loader = get_concatenated_dataloader(client_names, "train", args, params,num_workers=8)
    #train_loader = get_dataloader(client_names, "train", args, params,num_workers=8)
    #print("train_loader dataset len: ", len(train_loader.dataset))
    trainer = Trainer(args, params, data_path=client_names[0],validation_mode=True)
    #trainer.train_loader = PersistentDataLoader(train_loader)
    #temp solution
    #model_state = torch.load('/home/mattias/Documents/projects/YOLOv8-pt-fedn/results/lean_ml_complete_1juni/modelstates/modelstate_284')
    #trainer.model.load_state_dict(model_state)
    len_lu = len(trainer.train_loader.dataloader)
    args.local_updates = len_lu
    print("len_lu: ", len_lu)
    len_lu = len(trainer.val_loader.dataset)
    

    print("len_lu: ", len_lu)

    trainer.usage_counter = np.zeros(len(trainer.train_loader.dataloader.dataset), dtype=int)
    
    
   
    for epoch in range(0,2000):
        trainer.train()
        all_mean_map = []
        # raw model evalutation
        
           
        m_pre, m_rec, map50, mean_ap = trainer.validate(trainer.model,0.1)
        print("save_path: ", os.path.join(exp_name,client_names[0]+'_model_step.csv'))
        log_metrics(epoch, [m_pre, m_rec, map50, mean_ap], os.path.join(exp_name,name+'_model_step.csv'))
        all_mean_map += [mean_ap] 

            
        torch.save(trainer.model.state_dict(), os.path.join(weights_dir,"modelstate_"+str(epoch)))
        torch.save(trainer.ema.ema.state_dict(), os.path.join(weights_dir_ema,"emastate_"+str(epoch)))






if __name__ == "__main__":
    main()