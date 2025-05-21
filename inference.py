import os
from utils.dataset_wisard import Dataset
from torch.utils import data
import cv2
from matplotlib import pylab as plt
import numpy as np
import numpy as np
from fedn_util import load_weights_from_path_into_model
from nets import nn
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_with_boxes(image, target_tensor):
    """
    Plot a single image with bounding boxes using matplotlib.

    Args:
        image_tensor (torch.Tensor): Shape (3, H, W), values 0-255.
        target_tensor (torch.Tensor): Shape (n_boxes, 6), [:,1:] = (class, x_center, y_center, width, height) normalized.
    """
    #image = image_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    #image = image.transpose(1,2,0)
    h, w, _ = image.shape
    cc = 400
    fig, ax = plt.subplots(1, 1, figsize=(w/cc, h/cc))
    ax.imshow(image.astype('uint8'))

    for box in target_tensor:
        x1, y1, x2, y2,conf, classid = box.tolist()
        # från normaliserat till pixelkoordinater
        width, height = x2-x1, y2-y1
        x_min = x1
        y_min = y1

        rect = patches.Rectangle((x_min, y_min), width, height, 
                                  linewidth=2, linestyle='--', edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        #ax.text(x_min, y_min - 5, f'Class {int(cls)}', color='red', fontsize=12, backgroundcolor='white')
        
        label = f'{conf:.2f}'
        text_xc = np.maximum(np.minimum(x1,w),0)
        text_yc = np.maximum(np.minimum(y1 - 5,h),0)
        ax.text(text_xc, text_yc, label, color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    #plt.show()
    return fig, ax

def add_box(fig, ax, target_tensor, image_shape):
    """
    Plot a single image with bounding boxes using matplotlib.

    Args:
        image_tensor (torch.Tensor): Shape (3, H, W), values 0-255.
        target_tensor (torch.Tensor): Shape (n_boxes, 6), [:,1:] = (class, x_center, y_center, width, height) normalized.
    """
    #image = image_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    #image = image.transpose(1,2,0)
    h, w, _ = image_shape

    for box in target_tensor:
        classid, x1, y1, width, height = box.tolist()
        # från normaliserat till pixelkoordinater
        #width, height = (x2-x1)*w, (y2-y1)*h
        x_min = (x1-width/2)*w
        y_min = (y1-height/2)*h
        width *= w
        height *= h
        

        rect = patches.Rectangle((x_min, y_min), width, height, 
                                  linewidth=2, linestyle='--', edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        #ax.text(x_min, y_min - 5, f'Class {int(cls)}', color='red', fontsize=12, backgroundcolor='white')
        
        label = f'label'
        ax.text(x1, y1 - 5, label, color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))



    




def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
   

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)

def load_image(i, filenames):
        image = cv2.imread(filenames[i])[:,:,::-1]

        h, w = image.shape[:2]
        #r = input_size / max(h, w)
        #if r != 1:
        #    image = cv2.resize(image,
        #                       dsize=(int(w * r), int(h * r)),
        #                       interpolation=cv2.INTER_LINEAR)
        return image, (h, w)


def read_image(index, filenames, input_size):
        image, shape = load_image(index, filenames)
        h, w = image.shape[:2]
        # Resize
        sample, ratio, pad = resize(image, input_size)
        
        shapes = shape, ((h / shape[0], w / shape[1]), pad)  # for COCO mAP rescaling
        sample = sample.transpose((2, 0, 1))
        sample = np.ascontiguousarray(sample)
        return image, sample, ratio, pad

from utils import util
import torch
def inference(model,index,filenames, input_size, conf_threshold=0.5, iou_threshold=0.65):

    image, sample, ratio, pad  = read_image(index, filenames, input_size)
    sample = torch.from_numpy(sample).unsqueeze(0)
    sample = sample.cuda()
    sample = sample / 255
    #sample.device(device)
    with torch.no_grad():
        output = model(sample)
        output = util.non_max_suppression(output, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        for o in output:
            o[:,[0,2]] = (o[:,[0,2]] - pad[0]) /ratio[0]
            o[:,[1,3]] = (o[:,[1,3]] - pad[1])/ratio[1]    

    return output, image

def read_label(filename):
     
    with open(filename) as f:
        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
        label = np.array(label, dtype=np.float32)
    nl = len(label)
        
    if nl:
        assert label.shape[1] == 5, 'labels require 5 columns'
        assert (label >= 0).all(), 'negative label values'
        #assert (label[:, 1:] <= 1).all(), 'non-normalized coordinates'
        _, i = np.unique(label, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
            #print("duplicate, ", filename)
            #for r in label:
            #    print("  ", r)
            label = label[i]  # remove duplicates
            #print("nl: ", nl, "len label: ", len(label))

        x_center = label[:, 1]
        y_center = label[:, 2]
        w = label[:, 3]
        h = label[:, 4]

        x_min = x_center - w / 2
        x_max = x_center + w / 2
        y_min = y_center - h / 2
        y_max = y_center + h / 2

        # Mask to keep only boxes fully inside [0, 1]
        inside_mask = (x_min >= 0) & (x_max <= 1) & (y_min >= 0) & (y_max <= 1)
        label = label[inside_mask]

            
    else:
        label = np.zeros((0, 5), dtype=np.float32)
    
    return label


def main():
    
    
    # ladda in modellen
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.yolo_v8_n(1).to(device)
    model_path = "/home/mattias/Documents/projects/YOLOv8-pt-fedn/downloaded_models/all_clients_100mu_best_val_map50.npz"
     
    load_weights_from_path_into_model(model_path, model)
    #w = torch.load("results/modelstates/modelstate_443")
    #model.state_dict = w
    model.eval()


    image_folder = 'uppsalaoffice'
    image_folder = '/home/mattias/Documents/projects/Wisard_usecase/datasets/dataset_Airfield/valid/images'
    include_label = True
    filenames = sorted([os.path.join(image_folder, ex) for ex in os.listdir(image_folder)])
    input_size = 640

    os.makedirs("inference", exist_ok=True)
    conf_threshold = 0.1
    name_model_threshold = 'bestmap50_'+str(conf_threshold)+'_0.65'
    path = os.path.join("inference", name_model_threshold + "_" + "_".join(image_folder.split("/")[-3:-1]))
    os.makedirs(path, exist_ok=True)
    for i in range(len(filenames)):
        if i%10==0:
            print(i)
            torch.cuda.empty_cache()
        output, image = inference(model, i, filenames, input_size, conf_threshold)
        
        fig, ax = plot_image_with_boxes(image, output[0])
        
        rec = ".".join(filenames[i].split("/")[-1].split(".")[:-1]) + '.txt'
        
        filename = os.path.join("/".join(image_folder.split("/")[:-1]),"labels",rec)
        
        if include_label:
            label = read_label(filename)
            add_box(fig, ax, label, image.shape)
        #save_figure_to_file(fig, os.path.join(path, f"{i}.png"))
        plt.savefig(os.path.join(path, str(i)+'.png'))
        plt.close(fig)
        
        
if __name__ == "__main__":
    main()
