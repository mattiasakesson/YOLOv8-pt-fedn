import json
import os
import time

import numpy as np
import torch
import tqdm
from fedn.utils.helpers.helpers import save_metadata

from dataloader import get_dataloader
from fedn_util import extract_weights_from_model, load_weights_into_model
from nets import nn
from utils import util


class Trainer:
    def __init__(self, args, params):
        self.args = args
        self.params = params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.yolo_v8_n(len(params["names"].values())).to(self.device)
        self.ema = util.EMA(self.model)

        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

        data_path = os.getenv("DATA_PATH")
        torch.multiprocessing.set_start_method("spawn", force=True)
        self.train_loader = get_dataloader(data_path, "train", args, params)
        self.val_loader = get_dataloader(data_path, "valid", args, params)

        self.prev_iterations = 0

    def configure_optimizer(self):
        accumulate = max(round(64 / self.args.batch_size), 1)
        self.params["weight_decay"] *= self.args.batch_size * accumulate / 64
        p = [], [], []
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                p[2].append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):
                p[1].append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                p[0].append(v.weight)

        optimizer = torch.optim.Adam(p[2], self.params["lr0"])
        optimizer.add_param_group(
            {"params": p[0], "weight_decay": self.params["weight_decay"]}
        )
        optimizer.add_param_group({"params": p[1]})
        return optimizer

    def configure_scheduler(self):
        TOTAL_EPOCHS = 1000

        def lr(x):
            return (1 - x / TOTAL_EPOCHS) * (1.0 - self.params["lrf"]) + self.params[
                "lrf"
            ]

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr, last_epoch=-1)

    def lr_schedule(self, epoch):
        """Learning rate decay schedule."""
        TOTAL_EPOCHS = 1000  # self.args.epochs  # Or use a class constant if you prefer
        return (1 - epoch / TOTAL_EPOCHS) * (1.0 - self.params["lrf"]) + self.params[
            "lrf"
        ]

    def train(self, weights, client_setting):
        print("trainer train starts")
        t0 = time.time()
        old_weights =  [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        load_weights_into_model(weights, self.model)
        upd_weights =  [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        distance = np.sum([np.linalg.norm(a-b) for a,b in zip(old_weights,upd_weights)])
        print("train distance: ", distance)
        print("old state: ",  np.sum([np.linalg.norm(a) for a in old_weights]))
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        num_batch = len(self.train_loader)
        epochs = int(np.ceil(self.args.local_updates / num_batch))
        print("epochs: ", epochs)
        print("prev_itterations: ", self.prev_iterations)
        accumulate = max(round(64 / self.args.batch_size), 1)
        amp_scale = torch.amp.GradScaler("cuda")
        criterion = util.ComputeLoss(self.model, self.params)
        num_warmup = max(round(self.params["warmup_epochs"] * num_batch), 1000)
        num_warmup = 1000
        # print("num_warmup: ", num_warmup)

        updates = 0
        done = False
        warm_up = False
        for epoch in range(epochs):
            print("epoch: ", epoch, "/", epochs)
            self.model.train()

            if self.args.epochs - epoch == 10:
                self.train_loader.dataset.mosaic = False

            m_loss = util.AverageMeter()

            p_bar = enumerate(self.train_loader)

            print(("\n" + "%10s" * 5) % ("epoch", "memory", "warm_up", "x", "loss"))

            p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            self.optimizer.zero_grad()

            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch + self.prev_iterations  # number of iterations

                # print("i: ", i, "x: ", x)
                samples = samples.cuda().float() / 255
                targets = targets.cuda()

                # Warmup

                if x <= num_warmup:
                    warm_up = True
                    xp = [0, num_warmup]
                    fp = [1, 64 / self.args.batch_size]
                    accumulate = max(1, np.interp(x, xp, fp).round())
                    for j, y in enumerate(self.optimizer.param_groups):
                        y.setdefault("initial_lr", y["lr"])  # store if missing
                        if j == 0:
                            fp = [
                                self.params["warmup_bias_lr"],
                                y["initial_lr"] * self.lr_schedule(epoch),
                            ]
                        else:
                            fp = [0.0, y["initial_lr"] * self.lr_schedule(epoch)]
                        y["lr"] = np.interp(x, xp, fp)
                        if "momentum" in y:
                            fp = [
                                self.params["warmup_momentum"],
                                self.params["momentum"],
                            ]
                            y["momentum"] = np.interp(x, xp, fp)

                # Forward
                with torch.amp.autocast("cuda"):
                    outputs = self.model(samples)  # forward
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= self.args.batch_size  # loss scaled by batch_size
                # loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    # print("accumalate model")
                    amp_scale.unscale_(self.optimizer)  # unscale gradients
                    util.clip_gradients(self.model)  # clip gradients
                    amp_scale.step(self.optimizer)  # optimizer.step
                    amp_scale.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)

                # Log

                memory = f"{torch.cuda.memory_reserved() / 1e9:.3g}G"  # (GB)
                s = ("%10s" * 3 + "%10.4g" * 2) % (
                    f"{epoch + 1}/{self.args.epochs}",
                    memory,
                    str(warm_up),
                    x,
                    m_loss.avg,
                )
                p_bar.set_description(s)

                del loss
                del outputs
                self.prev_iterations += 1
                updates += 1
                if updates >= self.args.local_updates:
                    done = True
                    break

            # Scheduler
            self.scheduler.step()
            if done:
                break

        torch.cuda.empty_cache()
        # return ema.ema.state_dict()
        out_model = extract_weights_from_model(self.model)
        metadata = {
            "training_metadata": {
                # num_examples are mandatory
                "num_examples": 1,  # len(train_loader.dataset),
                "batch_size": self.train_loader.batch_size,
                "epochs": 1,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        }
        outpath = "temp"
        save_metadata(metadata, outpath)
        with open(outpath + "-metadata", "r") as fh:
            training_metadata = json.loads(fh.read())

        os.unlink(outpath + "-metadata")
        upd_weights =  [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        print("training done")
        t1 = time.time()
        print("time: ", t1 - t0)

        return out_model, training_metadata

    @torch.no_grad()
    def validate(self, weights):
        print(
            "validate start",
        )
        t0 = time.time()
        old_weights =  [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        load_weights_into_model(weights, self.model)
        upd_weights =  [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        distance = np.sum([np.linalg.norm(a-b) for a,b in zip(old_weights,upd_weights)])
        print("val distance: ", distance)
        print("old state: ",  np.sum([np.linalg.norm(a) for a in old_weights]))
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        #self.model.half()
        self.model.eval()

        # Configure
        iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        m_pre = 0.0
        m_rec = 0.0
        map50 = 0.0
        mean_ap = 0.0
        metrics = []
        p_bar = tqdm.tqdm(
            self.val_loader, desc=("%10s" * 3) % ("precision", "recall", "mAP")
        )
        for samples, targets, shapes in p_bar:
            samples = samples.cuda()
            targets = targets.cuda()
            #samples = samples.half()  # uint8 to fp16/32
            samples = samples / 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = samples.shape  # batch size, channels, height, width
            # Inference
            outputs = self.model(samples)

            # NMS
            targets[:, 2:] *= torch.tensor(
                (width, height, width, height)
            ).cuda()  # to pixels
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
                util.scale(
                    detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1]
                )

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
                            matches = torch.cat(
                                (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
                            )
                            matches = matches.cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[
                                    np.unique(matches[:, 1], return_index=True)[1]
                                ]
                                matches = matches[
                                    np.unique(matches[:, 0], return_index=True)[1]
                                ]
                            correct[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(
                        correct, dtype=torch.bool, device=iou_v.device
                    )
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        # Compute metrics
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

        # Print results
        print("%10.3g" * 3 % (m_pre, m_rec, mean_ap))

        # Return results
        self.model.float()  # for training

        performance = {
            "val_precision": m_pre,
            "val_recall": m_rec,
            "val_map50": map50,
            "val_map": mean_ap,
        }
        upd_weights =  [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        print("validation done")
        t1 = time.time()
        print("validation time: ", t1 - t0)
        return performance
