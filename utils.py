import numpy as np
import itertools,time
import sys, os
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import pickle    
import torchvision.models as models
import torch.nn as nn


def split_data(full_dataset, train_split=0.8, seed=1):   
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(full_dataset) == 1:
        train_dataset = test_dataset = full_dataset
    else:
    
        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def evaluate_data(model, datal, criterion, device, report=False):
    # Iterate over data.
    tsamples = 0.
    running_loss = 0.

    Y_pred = []
    Y_true = []
    with torch.no_grad():
        for inputs,_, labels in datal:

            inputs = inputs.to(device)
            labels = labels.to(device)

            try:
                outputs = model(inputs)
            except:
                import pdb;pdb.set_trace()

            _, preds = torch.max(outputs, 1)

            running_loss += criterion(outputs, labels).item()

            Y_pred.extend(preds)
            Y_true.extend(labels)

            tsamples += inputs.size(0)

    # Show confusion matrix for training/test samples
    Y_pred = np.array([int(x_) for x_ in Y_pred])
    Y_true = np.array([int(x_) for x_ in Y_true])

    print(Y_pred, Y_true)
    
    acc = np.mean(Y_pred==Y_true)
    loss = running_loss / tsamples 

    if report:
       # from sklearn.metrics import confusion_matrix
       # confusion_matrix(Y_true, Y_pred)
        from sklearn.metrics import classification_report
        print(classification_report(Y_true, Y_pred))
    return acc, loss , Y_pred, Y_true



def get_model(weight_dir, n_classes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft = models.vgg16(pretrained=True)

    num_ftrs = list(model_ft.classifier)[-1].in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.classifier[-1] = nn.Linear(num_ftrs, n_classes)
    
    state_dict = torch.load(weight_dir)
    
    model_ft.load_state_dict(state_dict)
    model_ft.eval()
    model_ft.to(device)
    
    return model_ft


def get_book_page(x):
    book  = x.split('/')[-3]
    page = x.split('/')[-1]
    page = page.replace('.jpg','').replace('p','')
    page_id = '{}_p{}'.format(book, str(int(page)+1))
    return book, int(page), page_id


def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def ensure_color_image(im):
    if len(np.shape(im))==2:
        im = np.stack([im,im,im], axis=2)
    assert np.shape(im)[2] == 3
    return im
