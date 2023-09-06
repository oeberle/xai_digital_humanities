import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch
import os
import pickle
    
from utils import get_model, set_up_dir
import torch.nn as nn
from xai import explain, get_prediction, plot_heatmap
from data_curation import get_data_10112022_sphaera_machines_boxes

seed=1
np.random.seed(seed)
torch.manual_seed(seed)

def main():

    ### Get Data ###
    data_root = 'sample_data/pages/'
    csv_file = 'sample_data/data_subset.csv'
    dataloader = sphaera_pages(data_root, csv_file, refsize=800, removemean=True, train_split=1., seed=seed, return_loader=True) 
    eval_loader = dataloader['train']
    weight_dir = 'model.pt' # <savedir>/.../model_train.pt
    model = get_model(weight_dir, n_classes=4)


    ### Collect explanations and save them for each class ###
    savedir='experiments/25112022_four_class_paper_data'
    set_up_dir(savedir)
    title_map = {0: 'other page', 1: 'instrument', 2: 'machine', 3: 'illustrations'}

    # Set up results directory for each class
    save_dir_map = {}
    for k,v in title_map.items():
        str_ = '{}_{}'.format(k,v)
        savedir_='{}/{}'.format(savedir, str_)        
        set_up_dir(savedir_)
        save_dir_map[k] = savedir_
        set_up_dir(savedir_)

    k=0
    Ls = []
    Rs = []
    data = {}

    for loader_data in  [dataloader['test'], dataloader['train']]:

        for x in loader_data:   

            for j in range(1):

                y_pred, logits, probs =  get_prediction(model, x[0], device)
                cases = [i for i in sorted(title_map.keys())]

                y_true = int(x[2][j])
                confidence = '{:0.4f}'.format(probs[y_true])

                save_dir_  = save_dir_map[y_true] 
                fstr = os.path.split(x[1][j])[-1].replace('.jpg', '')
                file_name = os.path.join(save_dir_, '{}_cl{}_{}.jpg'.format(confidence, y_true, fstr))

                if os.path.isfile(file_name):
                    print('exists', file_name.replace(save_dir_, ''))
                else:              
                    R_dict = {k: None for k in cases}
                    for l,cl  in enumerate(cases):
                        R_ = explain(model, x[0].to(device), cl=cl, device=device, n_classes=len(title_map))
                        Ls.append(logits[cl])
                        Rs.append(R_[0][0].detach().cpu().numpy().sum())
                        R = np.array(R_[0][0].detach().cpu().numpy()).sum(axis=0)
                        R_dict[cl] = R
                        print(cl, R[R>=0].sum(),  R[R<0].sum(),  logits[cl], probs[cl])

                    f, axs = plt.subplots(1,len(title_map), figsize=(8.,4.7))
                    b = np.max([10*((np.abs(R)**3.0).mean()**(1.0/3)) for R in R_dict.values()])
                     
                    for l, cl in enumerate(cases):
                        title_map_ = {k:v for k,v in title_map.items()}
                        title_map_[y_true] += '*'
                        plot_heatmap(R_dict[cl], x[0].squeeze()[0], (f,axs[l]), l, title_map_, cl, b)

                    f.tight_layout()
                    f.suptitle('predicted : {} - {:0.4f}'.format(title_map[y_pred], probs[y_pred]), y=0.99) #0.91)


                    if y_true !=-1:
                        f.savefig(file_name, dpi=250)
                    else:
                        f.savefig(os.path.join(save_dir_, '{}.jpg'.format(fstr)), dpi=250)

                    plt.cla()
                    plt.close()
                          
                    data[fstr] = {'R' : R_dict, 'y_true' : y_true, 'y_pred': y_pred, 'logits' : logits, 'probs': probs}
                if k==1:
                    break
                k+=1
                              
        pickle.dump(data, open(os.path.join(savedir, 'data.p'), 'wb'))

if __name__ == '__main__':
    main()