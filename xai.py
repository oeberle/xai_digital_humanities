import numpy
import copy
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from utils import set_up_dir
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy

def get_prediction(model, inputs, device):
    
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = nn.Softmax(dim=-1)(outputs) 
        vals, preds = torch.max(outputs, 1)
                             
    return int(preds.squeeze()), outputs.detach().cpu().numpy().squeeze(), probs.detach().cpu().numpy().squeeze()


def heatmap(R, fax = None):

    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))
    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    f,ax = fax
    
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    h=ax.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.colorbar(h)


def newlayer(layer,g):

    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer


def toconv(layers):
    # convert VGG classifier's dense layers to convolutional layers

    newlayers = []

    for i,layer in enumerate(layers):
        if isinstance(layer,nn.Linear):
            newlayer = None
            if i == 0:
                m,n = 512,layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))

            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]

    return newlayers


def explain(model, X, cl, device='cpu', n_classes=2):
    # compute LRP explanations
    # https://git.tu-berlin.de/gmontavon/lrp-tutorial 
    
    layers = list(model._modules['features']) +  toconv([model._modules['avgpool']]) +toconv(list(model._modules['classifier']))
    L = len(layers)
    
    A = [X]+[None]*L
    for l in range(L):
        A[l+1] = layers[l].forward(A[l])
        
    model_outputs = model(X)

    T = torch.FloatTensor((1.0*(numpy.arange(n_classes)==cl).reshape([1,n_classes,1,1]))).to(device)

    R = [None]*L + [(A[-1]*T).data]
        
    for l in range(1,L)[::-1]:
            
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
            
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d) or isinstance(layers[l],torch.nn.AdaptiveAvgPool2d) :
            
            if l <= 10:       rho = lambda p: p + 0.5*p.clamp(min=0);  incr = lambda z: z+1e-9
            if 11 <= l <= 17: rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9                
            if 18 <= l <= 24: rho = lambda p: p + 0.1*p.clamp(min=0);  incr = lambda z: z+1e-9
            if l > 24:       rho = lambda p: p;                        incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l],rho).forward(A[l]))        # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4

        else:
            R[l] = R[l+1]

    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(1,-1,1,1).to(device)
    std  = torch.Tensor([0.5, 0.5, 0.5]).reshape(1,-1,1,1).to(device)
            
    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)          # step 1 (b)
    z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)          # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data    
    
    return R



def plot_heatmap(R, x, fax, l, title_map, cl, b):

    f,ax = fax
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    # plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    ax.imshow(x, cmap='Greys')
    if l==0:
        ax.set_title('explained : {}'.format(title_map[cl]))
    else:
        ax.set_title('{}'.format(title_map[cl]))

    h=ax.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest', alpha=0.9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if l!=len(title_map)-1:
        cax.set_axis_off()
    else:
        plt.colorbar(h, cax=cax)
