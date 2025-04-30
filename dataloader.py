import os
from utils.dataset_wisard import Dataset
from torch.utils import data


def get_dataloader(client_name, set_, args, params,num_workers=8):
    filenames = []
    with open(os.path.join(client_name, set_+".txt")) as reader:
        for filename in reader.readlines():
            filenames.append(filename[:-1])

    #for f in filenames:
     #   print(f)
    dataset = Dataset(filenames, args.input_size, params, set_== "train")
    
    if set_ == 'train':
        batch_size = args.batch_size
    else:
        batch_size = 1

    loader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True, collate_fn=Dataset.collate_fn,
                             persistent_workers=True if num_workers > 0 else False, shuffle=True)
    
    return loader


def get_concatenated_dataloader(client_names, set_, args, params,num_workers=8):
    filenames = []
    for client_name in client_names:
        with open(os.path.join(client_name, set_+".txt")) as reader:
            for filename in reader.readlines():
                filenames.append(filename[:-1])

    #for f in filenames:
     #   print(f)
    print("len filenames: ", len(filenames))
    dataset = Dataset(filenames, args.input_size, params, set_== "train")
    print("dataset len: ", len(dataset))
    if set_ == 'train':
        batch_size = args.batch_size
    else:
        batch_size = 1

    loader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True, collate_fn=Dataset.collate_fn,
                             persistent_workers=True if num_workers > 0 else False, shuffle=True)
    
    return loader