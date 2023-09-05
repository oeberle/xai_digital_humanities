"""
This file contains helper methods to binarize an image. Taken from OCRopus: https://github.com/tmbdev/ocropy
"""

import numpy as np
import numpy
from numpy import (amax, amin, array, bitwise_and, clip, dtype, mean, minimum,
                   nan, sin, sqrt, zeros)
import matplotlib.pyplot as plt
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
import PIL
import pandas as pd
import torchvision.transforms as transforms
import torch
import os.path
from torchvision.datasets.utils import download_url
from utils import set_up_dir
import sys
import os


def save_and_proc_page(row, dst_dir):

    x = row.page_id
    url = row.pageIIIF

    book  = '_'.join(x.split('_')[:-1])
    page = x.split('_')[-1]
    page = page.replace('.jpg','').replace('p','')


    book_dir = os.path.join(dst_dir, book)

    proc_dir = os.path.join(book_dir, 'processed')
    raw_dir = os.path.join(book_dir, 'raw')

    file_name =  str(x) #'{}'.format(int(page)-1)+".jpg"

    if not os.path.exists(os.path.join(raw_dir, file_name)):
        download_url(url, root=raw_dir, filename=file_name, md5=None)

    file_path_bin = os.path.join(proc_dir, file_name)

    if not os.path.exists(file_path_bin):
        image = read_image_gray(os.path.join(raw_dir, file_name))
        image = binarize(image)
        write_image_binary(file_path_bin, image)

    print(file_path_bin)


def check_if_available(data, root_dir, return_missing=False):
    page_ids = data.page_id
    
    yes = []
    yes = [1 if os.path.isfile(page_id2fname(p, root_dir, case='processed')) else 0 for p in page_ids]
    data_filt = data[np.array(yes)==1]
    if return_missing:
        return data[np.array(yes)!=1]
    else:
        return data_filt

def get_book_page(x):
    book  = x.split('/')[-3]
    page = x.split('/')[-1]
    page = page.replace('.jpg','').replace('p','')
    page_id = '{}_p{}'.format(book, str(int(page)+1))
    return book, int(page), page_id


def page_id2fname(page_id, root_dir, case='processed'):

    book = '_'.join(page_id.split('_')[:-1])
    return os.path.join(root_dir, book, case, page_id)


def page_id2fname_old(page_id, root_dir, case='processed'):

    book = '_'.join(page_id.split('_')[:-1])
    page = page_id.split('_')[-1]

    page = page.replace('.jpg','').replace('p','')
    return os.path.join(root_dir, book, case, '{}.jpg'.format(int(page)-1))

def merge(csv1, csv2, extra_fac = 1):
    np.random.seed(1)
    
    df1 = pd.read_csv(csv1)
    page_ids1 = list(df1.page_id)
    y1 = [1]*len(page_ids1)
    xywhs1 = list(df1.xywh)
    
    df2 = pd.read_csv(csv2)
    page_ids2 = list(df2.page_id)
    
    np.random.shuffle(page_ids2)
    N2_before = len(page_ids2)
    N2_sampled = int(len(page_ids1)*extra_fac)
    
    print(len(page_ids1), N2_before, N2_sampled)

    page_ids2 = page_ids2[:N2_sampled]
    
    y2 = [0]*N2_sampled
    xywhs2 = list(df2.xywh)[:N2_sampled]
    
    page_ids = page_ids1 +  page_ids2
    y = y1 + y2
    xywhs = xywhs1 + xywhs2
    xywhs = [str(x) for x in xywhs]
        
    books = ['_'.join(x.split('_')[:-1]) for x in page_ids]
    page_nrs = [int(x.split('_')[-1].replace('.jpg','').replace('p','')) for x in page_ids]
    
    df  = pd.DataFrame(data = np.array([page_ids, page_nrs, books, y, xywhs]).T, columns=[ 'page_id', 'page_nr',  'book', 'label', 'xywh'])

    return df


def indiviual_resize(x, refsize=1200):
    shape = np.shape(x)
    h,w = shape[0], shape[1]
    ratio = float(h)/w
    if ratio >= 1.:
        h = refsize
        w = int(h/ratio)
    elif  ratio < 1.:
        w = refsize
        h = int(w*ratio)
    xout = transforms.Resize([h,w])(x)
    return xout

def randomaffine(x, rotate, translate, shear, scale):
    xmin = x.min()
    xmax = x.max()
    x_255 = ((x - xmin)/(xmax-xmin))
    xpil =  transforms.ToPILImage()(x_255)
    xpil = transforms.RandomAffine(rotate, translate= translate,shear=shear, scale=scale, fillcolor=0)(xpil)
    xarr = np.asarray(xpil)
    xback = torch.Tensor(xarr).unsqueeze(0)
    xback = (xback/255.)*(xmax-xmin) + xmin
    return xback



def normalize_raw_image(raw):
    ''' perform image normalization '''
    image = raw-np.amin(raw)
    if np.amax(image)==np.amin(image):
        print_info("# image is empty: %s" % (fname))
        return None
    image = image/np.amax(image)
    return image


def estimate_local_whitelevel(image, zoom=0.5, perc=80, range=20, debug=0):
    '''flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    '''
    m = interpolation.zoom(image,zoom)
    m = filters.percentile_filter(m,perc,size=(range,2))
    m = filters.percentile_filter(m,perc,size=(2,range))
    m = interpolation.zoom(m,1.0/zoom)
    if debug>0:
        plt.clf()
        plt.imshow(m,vmin=0,vmax=1)
        plt.ginput(1,debug)
    w,h = np.minimum(np.array(image.shape),np.array(m.shape))
    flat = np.clip(image[:w,:h]-m[:w,:h]+1,0,1)
    if debug>0:
        plt.clf()
        plt.imshow(flat,vmin=0,vmax=1)
        plt.ginput(1,debug)
    return flat

def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90, debug=0):
    '''# estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    '''
    d0,d1 = flat.shape
    o0,o1 = int(bignore*d0),int(bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est-filters.gaussian_filter(est,e*20.0)
        v = filters.gaussian_filter(v**2,e*20.0)**0.5
        v = (v>0.3*np.amax(v))
        v = morphology.binary_dilation(v,structure=np.ones((int(e*50),1)))
        v = morphology.binary_dilation(v,structure=np.ones((1,int(e*50))))
        if debug>0:
            plt.imshow(v)
            plt.ginput(1,debug)
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),lo)
    hi = stats.scoreatpercentile(est.ravel(),hi)
    return lo, hi


def binarize(raw, threshold=0.5):
	image = normalize_raw_image(raw)
	flat = estimate_local_whitelevel(image)
	lo, hi = estimate_thresholds(flat)
	flat -= lo
	flat /= (hi-lo)
	#print(lo,hi, flat)
	flat = np.clip(flat,0,1)
	bin = 1*(flat>threshold)
	return bin


def pil2array(im,alpha=0):
    if im.mode=="L":
        a = numpy.fromstring(im.tobytes(),'B')
        a.shape = im.size[1],im.size[0]
        return a
    if im.mode=="RGB":
        a = numpy.fromstring(im.tobytes(),'B')
        a.shape = im.size[1],im.size[0],3
        return a
    if im.mode=="RGBA":
        a = numpy.fromstring(im.tobytes(),'B')
        a.shape = im.size[1],im.size[0],4
        if not alpha: a = a[:,:,:3]
        return a
    return pil2array(im.convert("L"))


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def array2pil(a):
    if a.dtype==dtype("B"):
        if a.ndim==2:
            return PIL.Image.frombytes("L",(a.shape[1],a.shape[0]),a.tostring())
        elif a.ndim==3:
            return PIL.Image.frombytes("RGB",(a.shape[1],a.shape[0]),a.tostring())
        else:
            raise OcropusException("bad image rank")
    elif a.dtype==dtype('float32'):
        return PIL.Image.fromstring("F",(a.shape[1],a.shape[0]),a.tostring())
    else:
        raise OcropusException("unknown image type")

def isbytearray(a):
    return a.dtype in [dtype('uint8')]

def isfloatarray(a):
    return a.dtype in [dtype('f'),dtype('float32'),dtype('float64')]

def isintarray(a):
    return a.dtype in [dtype('B'),dtype('int16'),dtype('int32'),dtype('int64'),dtype('uint16'),dtype('uint32'),dtype('uint64')]

def isintegerarray(a):
    return a.dtype in [dtype('int32'),dtype('int64'),dtype('uint32'),dtype('uint64')]


def read_image_gray(fname,pageno=0):
    """Read an image and returns it as a floating point array.
    The optional page number allows images from files containing multiple
    images to be addressed.  Byte and short arrays are rescaled to
    the range 0...1 (unsigned) or -1...1 (signed)."""

    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    if a.dtype==dtype('uint8'):
        a = a/255.0
    if a.dtype==dtype('int8'):
        a = a/127.0
    elif a.dtype==dtype('uint16'):
        a = a/65536.0
    elif a.dtype==dtype('int16'):
        a = a/32767.0
    elif isfloatarray(a):
        pass
    else:
        raise OcropusException("unknown image type: "+a.dtype)
    if a.ndim==3:
        a = mean(a,2)
    return a


def write_image_gray(fname,image,normalize=0,verbose=0):
    """Write an image to disk.  If the image is of floating point
    type, its values are clipped to the range [0,1],
    multiplied by 255 and converted to unsigned bytes.  Otherwise,
    the image must be of type unsigned byte."""
    if verbose: print("# writing", fname)
    if isfloatarray(image):
        image = array(255*clip(image,0.0,1.0),'B')
    assert image.dtype==dtype('B'),"array has wrong dtype: %s"%image.dtype
    im = array2pil(image)
    im.save(fname)


def read_image_binary(fname,dtype='i',pageno=0):
    """Read an image from disk and return it as a binary image
    of the given dtype."""
    if type(fname)==tuple: fname,pageno = fname
    assert pageno==0
    pil = PIL.Image.open(fname)
    a = pil2array(pil)
    if a.ndim==3: a = amax(a,axis=2)
    return array(a>0.5*(amin(a)+amax(a)),dtype)


def write_image_binary(fname,image,verbose=0):
    """Write a binary image to disk. This verifies first that the given image
    is, in fact, binary.  The image may be of any type, but must consist of only
    two values."""
    if verbose: print("# writing", fname)
    assert image.ndim==2
    image = array(255*(image>midrange(image)),'B')
    im = array2pil(image)
    im.save(fname)
    
    return None


def midrange(image,frac=0.5):
    """Computes the center of the range of image values
    (for quick thresholding)."""
    return frac*(amin(image)+amax(image))

