import os
print(f"Process PID: {os.getpid()}")
import collections
from importlib import util
import io
import json
import os
import uuid
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import collections
import copy
from wisard_fl import train, validate
import argparse
import torch
import os
from utils import util
import yaml
from nets import nn
from utils import util
from utils.dataset_wisard import Dataset
from torch.utils import data
import time

from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient
from fedn.utils.helpers.helpers import get_helper, save_metadata


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
            print("filenotexists")
            writer.writeheader()
        writer.writerow({
            'round': str(round_id + 1).zfill(3),
            'precision': f'{metrics[0]:.3f}',
            'recall': f'{metrics[1]:.3f}',
            'mAP@50': f'{metrics[2]:.3f}',
            'mAP': f'{metrics[3]:.3f}'
        })
        f.flush()



TOTAL_EPOCHS = 1000
def learning_rate(args, params):
    def fn(x):
        return (1 - x / TOTAL_EPOCHS) * (1.0 - params['lrf']) + params['lrf']

    return fn

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

parser = argparse.ArgumentParser()
parser.add_argument('--input-size', default=640, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--epochs', default=500, type=int)

args = parser.parse_args()

#util.setup_seed()


with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
    params = yaml.safe_load(f)

# Dataloader
from dataloader import get_dataloader
data_path = os.getenv('DATA_PATH')
torch.multiprocessing.set_start_method("spawn", force=True)
train_loader = get_dataloader(data_path,'train', args,params)
val_loader = get_dataloader(data_path,'valid', args,params)


model = nn.yolo_v8_n(len(params['names'].values())).cuda()
ema = util.EMA(model)
# Optimizer
accumulate = max(round(64 / args.batch_size ), 1)
params['weight_decay'] *= args.batch_size * accumulate / 64

p = [], [], []
for v in model.modules():
    if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
        p[2].append(v.bias)
    if isinstance(v, torch.nn.BatchNorm2d):
        p[1].append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
        p[0].append(v.weight)

#optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)
optimizer = torch.optim.Adam(p[2], params['lr0'])

optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
optimizer.add_param_group({'params': p[1]})
del p

# Scheduler
lr = learning_rate(args, params)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

local_updates = 25
prev_iterations = 0

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



local_rank = int(os.getenv('LOCAL_RANK', 0))
world_size = int(os.getenv('WORLD_SIZE', 1))
print("local_rank: ", local_rank)
print("world_size: ", world_size)

@torch.no_grad()
def validate():
   
    print("validate start",)
   

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(val_loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for samples, targets, shapes in p_bar:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width
        # Inference
        outputs = model(samples)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            # SKYDD mot None-värden
            if samples[i] is None:
                
                print("samples[i]: ", samples[i])
                continue

            

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            if shapes[i] is None:

                print("shapes[i] is None: ", shapes[i])

                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return m_pre, m_rec, map50, mean_ap



def train():#in_model):
    global prev_iterations
    
    print("model train start")
    #model.load_state_dict(in_model)

    # Start training
    
    num_batch = len(train_loader)
    epochs = int(np.ceil(local_updates/num_batch))
    print("local_updates: ", local_updates)
    print("epochs: ", epochs)
    print("prev_itterations: ", prev_iterations)
    accumulate = max(round(64 /args.batch_size ), 1)
    print("accumulate: ", accumulate)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    num_warmup = 1000
    #print("num_warmup: ", num_warmup)
    
    updates = 0
    done = False
    warm_up = False
    for epoch in range(epochs):
        print("epoch: ", epoch, "/", epochs)
        model.train()

        if args.epochs - epoch == 10:
            train_loader.dataset.mosaic = False

        m_loss = util.AverageMeter()
        
        p_bar = enumerate(train_loader)
        
        print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'warm_up', 'x', 'loss'))
        
        p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

        optimizer.zero_grad()

        for i, (samples, targets, _) in p_bar:
            x = i + num_batch * epoch + prev_iterations  # number of iterations

            #print("i: ", i, "x: ", x)
            samples = samples.cuda().float() / 255
            targets = targets.cuda()

            # Warmup
            
            if x <= num_warmup:
                warm_up = True
                xp = [0, num_warmup]
                fp = [1, 64 /args.batch_size ]
                accumulate = max(1, np.interp(x, xp, fp).round())
                for j, y in enumerate(optimizer.param_groups):
                    if j == 0:
                        fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                    else:
                        fp = [0.0, y['initial_lr'] * lr(epoch)]
                    y['lr'] = np.interp(x, xp, fp)
                    if 'momentum' in y:
                        fp = [params['warmup_momentum'], params['momentum']]
                        y['momentum'] = np.interp(x, xp, fp)

            # Forward
            with torch.cuda.amp.autocast():
                outputs = model(samples)  # forward
            loss = criterion(outputs, targets)

            m_loss.update(loss.item(), samples.size(0))

            loss *= args.batch_size  # loss scaled by batch_size
            #loss *= args.world_size  # gradient averaged between devices in DDP mode

            # Backward
            amp_scale.scale(loss).backward()

            # Optimize
            if x % accumulate == 0:
                #print("accumalate model")
                amp_scale.unscale_(optimizer)  # unscale gradients
                util.clip_gradients(model)  # clip gradients
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Log
            
            memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
            s = ('%10s' * 3 + '%10.4g'*2) % (f'{epoch + 1}/{args.epochs}', memory, str(warm_up), x, m_loss.avg)
            p_bar.set_description(s)

            del loss
            del outputs
            prev_iterations += 1
            updates += 1
            if updates >= local_updates:
                done = True
                break
        

        # Scheduler
        scheduler.step()
        if done:
            break 


            

    #if args.local_rank == 0:
    #    util.strip_optimizer('./weights/best.pt')  # strip optimizers
    #    util.strip_optimizer('./weights/last.pt')  # strip optimizers
    print("training done")
    torch.cuda.empty_cache()
    #return ema.ema.state_dict()
    out_model = extract_weights_from_model()
    metadata = {
            "training_metadata": {
                # num_examples are mandatory
                "num_examples": 1,#len(train_loader.dataset),
                "batch_size": train_loader.batch_size,
                "epochs": 1,
                "lr": optimizer.param_groups[0]["lr"],
            }
        }
    outpath = "temp"
    save_metadata(metadata, outpath)
    with open(outpath + "-metadata", "r") as fh:
        training_metadata = json.loads(fh.read())

    os.unlink(outpath + "-metadata")
    return out_model, training_metadata




def validate_weights(weights):
    t0 = time.time()
    print("validatate_weights")
    
    load_weights_into_model(weights)

   
    m_pre, m_rec, map50, mean_ap = validate()
    # JSON schema
    performance = {
        "val_precision": m_pre,
        "val_recall": m_rec,
        "val_map50": map50,
        "val_map": mean_ap
    }
    print("validation done")
    t1 = time.time()
    print("validation time: ", t1-t0)
    return performance

def training_local():

    print("device: ", device)

    print("train_loader size: ", len(train_loader.dataset))

    for itteration in range(100):
        t0 = time.time()
        print("itteration: ", itteration)
        train()
        m_pre, m_rec, map50, mean_ap = validate()
        print("precision: ", np.round(m_pre,3))
        print("recall: ", np.round(m_rec,3))
        print("map50: ", np.round(map50,3))
        print("map: ", np.round(mean_ap,3))
        log_metrics(itteration, [m_pre, m_rec, map50, mean_ap], filename='local_results2.csv')
        t1 = time.time()
        print("time: ", t1-t0)



def training_round(weights, client_settings):
    # Convert from numpy array to correct pytorch format
    t0 = time.time()
    load_weights_into_model(weights)

    # Training loop
    local_epochs = 1
    train()

    metadata = {
        "training_metadata": {
            # num_examples are mandatory
            "num_examples": len(train_loader.dataset),
            "batch_size": train_loader.batch_size,
            "epochs": local_epochs,
            "lr": 0.001,# optimizer.param_groups[0]["lr"],
        }
    }

    out_model = extract_weights_from_model()

    outpath = "temp"
    save_metadata(metadata, outpath)
    with open(outpath + "-metadata", "r") as fh:
        training_metadata = json.loads(fh.read())

    os.unlink(outpath + "-metadata")
    t1 = time.time()
    print("training time: ", t1-t0)
    return out_model, training_metadata


def load_weights_into_model(weights):
    inpath = helper.get_tmp_path()
    with open(inpath, "wb") as fh:
        fh.write(weights.getbuffer())
    weights = helper.load(inpath)
    os.unlink(inpath)
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)


def extract_weights_from_model():
    # Convert from pytorch weights format numpy array
    updated_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    outpath = helper.get_tmp_path()
    helper.save(updated_weights, outpath)
    with open(outpath, "rb") as fr:
        out_model = io.BytesIO(fr.read())
    os.unlink(outpath)

    return out_model

def main():
    #project_url = "api.fedn.scaleoutsystems.com/testapi-rhe-fedn-reducer"
    project_url = os.getenv("PROJECT_URL")
    print("project_url: ", project_url)
    #client_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ0NTQyMjYzLCJpYXQiOjE3NDE5NTAyNjMsImp0aSI6IjVhNjQ4ZWNiNTk0YzQ0YTM5YzVkZDgyY2EzNjMwMWVjIiwidXNlcl9pZCI6MjE0LCJjcmVhdG9yIjoiTWF0dGlhcyIsInJvbGUiOiJjbGllbnQiLCJwcm9qZWN0X3NsdWciOiJ0ZXN0YXBpLXJoZSJ9.uwWENDYfGq_FKk3tQjcaDZNix_bZQlT0h1TWohvkSlg"
    client_token = os.getenv("FEDN_AUTH_TOKEN")
    print("client_token: ", client_token)
    fedn_client = FednClient(train_callback=training_round, validate_callback=validate_weights)

    name = "Tora"

    fedn_client.set_name(name)

    # client_id = str(uuid.uuid4())
    client_id = "214"
    fedn_client.set_client_id(client_id)
    print(client_id)
    controller_config = {
        "name": name,
        "client_id": client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = fedn_client.connect_to_api("https://" + project_url + "/", client_token, controller_config)

    print("result: ", result)
    print(combiner_config)
    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        exit(1)

    result: bool = fedn_client.init_grpchandler(config=combiner_config, client_name=name, token=client_token)
    print("result: ", result)
    if not result:
        exit(1)

    fedn_client.run()


if __name__ == "__main__":
    main()
    #training_local()
