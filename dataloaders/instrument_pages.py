from torch.utils.data import Dataset
import torch
from dataloaders.dataset_utils import LimitDataset, padding, crop, remove_mean, ToTensor
import torchvision.transforms as transforms
import os
from dataloaders.proc_utils import indiviual_resize
import pandas as pd
from PIL import Image
from dataloaders.dataset import PageDataset, PageDatasetBoxes
from utils import split_data

    

def sphaera_instruments_page(data_root, source, binarize=False, refsize=1200, removemean=True, train_split=0.8, seed=1, return_loader=True, extra_data = None, loader = PageDataset ) :   

    trans_list = [transforms.Grayscale(),
                  transforms.Lambda(lambda x: ToTensor(x)),
                  transforms.Normalize([0.5], [0.5]),]
    
    if removemean:
        trans_list = trans_list + [transforms.Lambda(lambda x: remove_mean(x))]

    extra_trans = []
    if refsize:
        extra_trans = [transforms.Lambda(lambda x: indiviual_resize(x, refsize=refsize))]
   
    page_trans =  transforms.Compose(extra_trans + trans_list) 
    
    data = loader(source=source,
                    root_dir = data_root,
                    transform = page_trans)
    
    train_dataset, test_dataset = split_data(data, train_split=train_split, seed=seed)
        
    train_data = torch.utils.data.ConcatDataset([train_dataset]+extra_data['train']) if extra_data is not None else train_dataset
    test_data = torch.utils.data.ConcatDataset([test_dataset]+ extra_data['test']) if extra_data is not None else test_dataset

    if return_loader:
    
        train_loader = torch.utils.data.DataLoader(
                train_data,
                 batch_size=1,
                 shuffle=True,
                 num_workers=2)


        test_loader = torch.utils.data.DataLoader(
                 test_data,
                 batch_size=1,
                 shuffle=False,
                 num_workers=2)

        dataloaders = {'train': train_loader,
                      'test': test_loader}

    else:

        dataloaders = {'train': train_data,
                       'test': test_data}

    
    return dataloaders
    
    
    
def sphaera_instruments(data_root, source, binarize=False, refsize=1200, removemean=True, train_split=0.8) :   

    trans_list = [transforms.Grayscale(),
                #  transforms.ToTensor(), 
                  transforms.Lambda(lambda x: ToTensor(x)),
                  transforms.Normalize([0.5], [0.5]),]
    
    if removemean:
        trans_list = trans_list + [transforms.Lambda(lambda x: remove_mean(x))]

    extra_trans = []
    if refsize:
        extra_trans = [transforms.Lambda(lambda x: indiviual_resize(x, refsize=refsize))]
   
    page_trans =  transforms.Compose(extra_trans + trans_list)
     
    data = PageDataset(source=source,
                    root_dir = data_root,
                    transform = page_trans)
    
    full_dataset = data
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(
             torch.utils.data.ConcatDataset([train_dataset, machine_train]),
             batch_size=1,
             shuffle=True,
             num_workers=2)
    
        
    test_loader = torch.utils.data.DataLoader(
             torch.utils.data.ConcatDataset([test_dataset, machine_test]), 
             batch_size=1,
             shuffle=False,
             num_workers=2)
    
    
    dataloaders = {'train': train_loader,
                  'test': test_loader}
    
    return dataloaders

    
   

