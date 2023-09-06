import torch
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
from tqdm import tnrange, tqdm_notebook, tqdm
from PIL import Image
import copy
import time
from utils import set_up_dir
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
    

def get_confusion_fig(y1,y2):
    cm = confusion_matrix(y1,y2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1,1,figsize=(8,8))
    pcm = axes.imshow(cm)
    fig.colorbar(pcm, ax=axes)
    return fig


class Trainer(object):
    def __init__(self, savedir, name,descr_name, epochs, weight_dir=None, eval_loader=None, model=None, jobdict=None, seed=0, lr_params=None, default_init_lr = 0.001):
        
        if weight_dir:
            self.start_epoch =  int(os.path.split(weight_dir)[1].split('_ep')[1].replace('.pt','')) + 1
        else:
            self.start_epoch = 0
        self.num_epochs =  self.start_epoch + epochs
        self.save_dir = savedir
        self.weight_dir = weight_dir
        self.name = name
        self.descr_name = descr_name
        self.save_dir_name = os.path.join(self.save_dir, self.name)
        set_up_dir(self.save_dir_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_loader = eval_loader
        self.model = model
        if jobdict:
            pfile = os.path.join(self.save_dir_name, 'jobdict.p')
            pickle.dump(jobdict, open(pfile, 'wb'))

        self.lr_params = lr_params
        self.default_init_lr = default_init_lr

        torch.manual_seed(seed)
        print('start training from', self.weight_dir, self.start_epoch)

        
        
    def evaluate_data(self, model, datal, criterion):
    

        # Iterate over data.
        tsamples = 0.
        running_loss = 0.
        
        Y_pred = []
        Y_true = []
        with torch.no_grad():
            for inputs,_, labels in datal:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_loss += criterion(outputs, labels).item()

                Y_pred.extend(preds)
                Y_true.extend(labels)

                tsamples += inputs.size(0)
                

        # Show confusion matrix for training/test samples
        Y_pred = np.array([int(x_) for x_ in Y_pred])
        Y_true = np.array([int(x_) for x_ in Y_true])

        acc = np.mean(Y_pred==Y_true)

        loss = running_loss / tsamples 
        return acc, loss
        
    def train(self, dataloader):
        
        if self.weight_dir:
            #Load weights
            self.model.load_state_dict(torch.load(self.weight_dir))
                 
        self.model.to(self.device)
        print('Only classifier params')
        params = self.model.classifier.parameters()                                         
        criterion = torch.nn.CrossEntropyLoss()
   
        if self.lr_params is not None:
        
            lr, step_size, gamma = self.lr_params[0], self.lr_params[1], self.lr_params[2]
            print('lr params', lr, step_size, gamma)
            optimizer = torch.optim.Adam(params, lr = lr)
            lr_scheduler_train = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        else:
            optimizer = torch.optim.Adam(params, lr = self.default_init_lr)
            lr_scheduler_train = None #lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
        model_trained, res_dict = self.train_model(self.model, dataloader, criterion, optimizer, scheduler=lr_scheduler_train)
    
        weight_file = '{}/final_model.pt'.format(self.save_dir_name)
        
        torch.save(model_trained.state_dict(), weight_file)
        pickle.dump(res_dict, open(os.path.join(self.save_dir_name, 'res_{}.p'.format(self.name)), 'wb'))
                 
        return weight_file
            
    def train_model(self, model, dataloader, criterion, optimizer, scheduler=None):

    
        dataset_sizes = {'train':  len(dataloader['train']),
                'test': len(dataloader['test']), 
                }

        print(dataset_sizes)
        writer = SummaryWriter(os.path.join(self.save_dir_name, self.descr_name))

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())

        res_dict = {'train': {'loss':[], 'acc' :[]},
                    'test': {'loss':[],  'acc' :[]},
                     'dataset_sizes': dataset_sizes}

        
        for epoch in tnrange(self.start_epoch, self.num_epochs, desc='Epochs'):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                Y_pred, Y_true = [],[]

                if phase == 'train':
                    if scheduler:
                        print('LR Scheduler step')
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_acc = 0.0
                running_corrects = 0

                # Iterate over data.
                it = 0
               
                datal = dataloader[phase] 
                tsamples = 0.

                    
                for inputs,_, labels in tqdm(datal, desc='Sample Loop'):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):


                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                       
                        Y_pred.extend(preds)
                        Y_true.extend(labels)


                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            #import pdb; pdb.set_trace()
                            optimizer.step()
                           

                    # statistics
                    running_loss += loss.item() #* inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    
                    tsamples += inputs.size(0)

                    it += 1
                    


                epoch_loss = running_loss / tsamples #dataset_sizes[phase]
                epoch_acc = running_corrects.double() /  tsamples

                res_dict[phase]['loss'].append(epoch_loss)
                res_dict[phase]['acc'].append(epoch_acc)

                #### Tensorboard:

                # write current loss
                writer.add_scalar(phase+'/epoch_loss', epoch_loss, epoch)
                writer.add_scalar(phase+'/epoch_acc', epoch_acc, epoch)

            
                # Show confusion matrix for training/test samples
                Y_pred = np.array([int(x_) for x_ in Y_pred])
                Y_true = np.array([int(x_) for x_ in Y_true])
                
                   
                
                acc_patches = np.mean(Y_pred==Y_true)
                res_dict[phase]['acc'].append(acc_patches)
                                
                fig = get_confusion_fig(Y_pred,Y_true)
                writer.add_figure('Patch Confusion ' + phase, fig, global_step=epoch)
                writer.add_scalar(phase+'/acc_patches', acc_patches, epoch)

                print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # deep copy the model
                if epoch == self.start_epoch:
                    best_loss = epoch_loss 


                if phase == 'test' and epoch_loss < best_loss and epoch>self.start_epoch:
                    #best_acc = epoch_acc
                    best_loss = epoch_loss 
                    best_model_wts = copy.deepcopy(model.state_dict())

                if phase == 'test':
                    res_dict['test']['loss'].append(epoch_loss)
                    torch.save(model.state_dict(), '{}/model_train.pt'.format(self.save_dir_name))
                   # if epoch%1==0:
                    torch.save(model.state_dict(), '{}/model_train_ep{}.pt'.format(self.save_dir_name, epoch))

                                       

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, res_dict
