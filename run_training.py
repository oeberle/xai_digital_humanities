import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import os
from trainer import Trainer
import torchvision.models as models
import torch.nn as nn
from dataloaders.sample_loader import sphaera_pages
    
seed=1
np.random.seed(seed)
torch.manual_seed(seed)

# Define Model
n_classes = 4
model_ft = models.vgg16(pretrained=True)
num_ftrs = list(model_ft.classifier)[-1].in_features
model_ft.classifier[-1] = nn.Linear(num_ftrs, n_classes)


# Load training data (here a small subset only)
data_root = 'sample_data/pages/'
csv_file = 'sample_data/data_subset.csv'
dataloader = sphaera_pages(data_root, csv_file, refsize=800, removemean=True, train_split=0.8, seed=seed, return_loader=True) 

dataset_sizes = {'train':  len(dataloader['train']),
    'test': len(dataloader['test'])}

print(dataset_sizes)

# Define and start Training
default_init_lr = 0.001
lr_params= (default_init_lr, 7, 0.1) 
epochs=30
descr_name=''
name='test_page'
savedir='experiments/0001_test_sample_data'
weight_dir=None

trainer = Trainer(savedir, name, descr_name, epochs, weight_dir=weight_dir, model = model_ft,  jobdict = None, seed = seed, lr_params= lr_params, default_init_lr = default_init_lr)
weight_file = trainer.train(dataloader)









