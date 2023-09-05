import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import os
from PIL import Image
import abc

from torch import tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import logging
import random
import torchvision
import pickle
from skimage import io

import os
from dataloaders.proc_utils import read_image_gray, binarize, write_image_binary
from utils import set_up_dir
import glob    
    
class PageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, source, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
#        super(PageDataset, self).__init__()



        if isinstance(source, str):
            df = pd.read_csv(source)
        elif isinstance(source, pd.core.frame.DataFrame):
            df = source
        else:
            raise
            
        self.pages = list(df.page_id)
        
        if 'label' in df:
            self.labels = list(df.label)
        else: 
            self.labels = [-1]*len(self.pages)
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.case = 'processed'
        
        self.missing = []
      
        
    def load_image(self, path):
        """
        Convenience method to load a page as PIL image. 
        
        Page is converted to three channel RGB image.
        """
        
        
        page = io.imread(path)
        page = Image.fromarray(page).convert('RGB') 
        page = transforms.Grayscale(num_output_channels=3)(page)
        return page

        
    def get_image(self,page_id):
        book = '_'.join(page_id.split('_')[:-1])
        fname = os.path.join(self.root_dir, book, self.case, page_id)
        return fname
 

    def __getitem__(self, idx):

       # print(idx)

        page = self.pages[idx]  
        y = self.labels[idx]

        img_name = self.get_image(page)
        page = self.load_image(img_name)

        if self.transform:
            page = self.transform(page)
            
            if page.shape[0]==1:
                page = page.repeat(3,1,1)

        return page, img_name, int(y)
    
    def __len__(self):
        return len(self.pages)
    
                  

class PageDatasetBoxes(PageDataset):
    
    def __init__(self, source, root_dir, transform=None, patch_transform=None, subset=None, pages=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        if isinstance(source, str):
            df = pd.read_csv(source)
        elif isinstance(source, pd.core.frame.DataFrame):
            df = source
        else:
            raise
            
        if pages:
            self.pages = pages
        else:
            self.pages = list(df.page_id)

        self.xywhs = [self.decode_box(x) for x in  list(df.xywh)]

                
        if subset:
            self.pages = [self.pages[i] for i in subset]
            self.xywhs = [self.xywhs[i] for i in subset]
        
        if 'label' in df:
            self.labels = list(df.label)
        else: 
            self.labels = [-1]*len(self.pages)
            
            
        self.root_dir = root_dir
        self.root_dir_sphaera = None # '/home/space/sphaera/sphaera_012020_all/pages/'

        self.transform = transform
        self.patch_transform = patch_transform
        
        self.case = 'processed'
        self.pad = 10
        
        self.missing = []
        
    @staticmethod
    def rescale_bbox(bbox, bbox_factors):
        return [int(bbox_factors[j]*x) for j,x in enumerate(bbox)]
    @staticmethod
    def decode_box(bbox):
        if bbox == 'nan':
            return bbox
        else:
            bbox = bbox.replace('[','').replace(']','')
            return [int(float(x)) for x in bbox.split(",")]

    def get_image(self,page_id):
        # .../data/1980_triegler_tractaetlein_1622/raw/1980_triegler_tractaetlein_1622_p181.jpg
        
        book = '_'.join(page_id.split('_')[:-1])
        page = page_id.split('_')[-1]
        page = page.replace('.jpg','').replace('p','')
       
        fname  = os.path.join(self.root_dir, book, self.case, page_id)
        
        return fname
 
        
    def __getitem__(self, idx):
        page = self.pages[idx]  
        y = self.labels[idx]
        bbox = self.xywhs[idx]

        img_name = self.get_image(page)
        page = self.load_image(img_name)
        page0=page

        
        if self.transform:
            h0, w0, _ = np.shape(page)
            page = self.transform(page)
            _, h1, w1 = np.shape(page)
            
            factor_h = float(h1)/h0
            factor_w = float(w1)/w0 
            
        bbox_factors = [factor_w, factor_h, factor_w, factor_h]
        bbox = self.rescale_bbox(bbox, bbox_factors)
        fi = bbox

        pad = self.pad 

                    
        ymin = int(int(fi[1])-pad) if  int(int(fi[1])-pad) > 0 else 0
        ymax = int(fi[1])+int(fi[3])+pad if int(fi[1])+int(fi[3])+pad <= h1 else h1
        
        xmin = int(fi[0])-pad  if  int(int(fi[0])-pad) > 0 else 0
        xmax = int(fi[0])+int(fi[2])+pad if int(fi[0])+int(fi[2])+pad <= w1 else w1
        

        try:
            page = page[:,ymin:ymax, 
                          xmin:xmax] 
        except:
            import pdb;pdb.set_trace()

        if self.patch_transform:
            page = self.patch_transform(page)
        
        if page.shape[0]==1:
            page = page.repeat(3,1,1)

        # pad small images
        if (np.array(page.shape[1:]) < 56).any()==True:
       
            print(page.shape)
            page = torch.nn.functional.pad(page, value=0, 
                                           pad=(int(np.ceil(np.clip(56-page.shape[2], a_min=0, a_max=None)/2)),
                                                int(np.ceil(np.clip(56-page.shape[2],a_min=0, a_max=None)/2)),
                                                int(np.ceil(np.clip(56-page.shape[1], a_min=0, a_max=None)/2)),
                                                int(np.ceil(np.clip(56-page.shape[1],a_min=0, a_max=None)/2))))#.shape

        return  page, img_name, int(y)
    
        
    
class ExternalPages(PageDataset):
    
    def __init__(self, root_dir, source = None, subset=None, label=-1, pages = None, transform=None):
 
        if pages:
            self.pages = pages
        else:
            self.pages = sorted(glob.glob( os.path.join(root_dir, '*.jpg')))
        
        if subset:
            self.pages = [self.pages[i] for i in subset]
 
        label = list(set( list(source.label)))
        assert len(label)==1

        self.labels = [label[0]]*len(self.pages)
        self.root_dir = root_dir
        self.transform = transform
                               
                               
    def get_image(self,page_id):
        return page_id        
    
class ExternalPagesBoxes(PageDatasetBoxes):
    
    def __init__(self, source, root_dir,  transform=None, patch_transform=None, subset=None, pages = None):
        super().__init__(source, root_dir, transform, patch_transform, subset, pages )

    def get_image(self,page_id):  
        return page_id      
 
