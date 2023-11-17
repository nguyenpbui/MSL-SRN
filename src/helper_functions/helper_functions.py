import os
import cv2
import time
import random
import pandas as pd
import numpy as np
import scipy
import scipy.io
from PIL import Image
from copy import deepcopy
from skimage import io, transform
from PIL import ImageFile
from PIL import ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision.transforms as transforms
from torchvision import datasets as datasets
from torch.utils.data import Dataset
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, roc_auc_score, classification_report


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def prf_cal(y_pred,y_true,k):
    """
    function to calculate top-k precision/recall/f1-score
    y_true: 0 1
    """
    GT=np.sum(y_true[y_true==1.])
    instance_num=y_true.shape[0]
    prediction_num=instance_num*k

    sort_indices = np.argsort(y_pred)
    sort_indices=sort_indices[:,::-1]
    static_indices = np.indices(sort_indices.shape)
    sorted_annotation= y_true[static_indices[0],sort_indices]
    top_k_annotation=sorted_annotation[:,0:k]
    TP=np.sum(top_k_annotation[top_k_annotation==1.])
    recall=TP/GT
    precision=TP/prediction_num
    f1=2.*recall*precision/(recall+precision)
    return precision, recall, f1


def evaluation_cal_multilabel(y_true, y_pred, num_classes=4, is_train=True):
    N, C = y_true.shape
    pred_extended = np.c_[y_pred, np.zeros(N)]
    true_extended = np.c_[y_true, np.zeros(N)]

    pred_extended[:,-1] = 1 - np.max(y_pred, axis=1)
    true_extended[:,-1] = 1 - np.max(y_true, axis=1)

    output = multilabel_confusion_matrix(true_extended, pred_extended)
    patient_level_acc = accuracy_score(true_extended, pred_extended)
    print(patient_level_acc)
    acc, sen, spe = [], [], []
    tps, tns, fps, fns = 0, 0, 0, 0
    for i in range(num_classes):
        tp, tn, fp, fn = output[i,1,1], output[i,0,0], output[i,0,1], output[i,1,0]
        # print(tp, tn, fp, fn)
        acc.append((tp+tn)/(tp+tn+fp+fn))
        sen.append(tp/(tp+fn))
        spe.append(tn/(tn+fp))
        tps += tp
        tns += tn
        fps += fp
        fns += fn
    # print(tps, tns, fps, fns)
    print((tps+tns)/(tps+tns+fps+fns), tps/(tps+fns), tns/(tns+fps))
    auc = roc_auc_score(true_extended, pred_extended, average=None)
    if not is_train:
        return acc, sen, spe, sum(acc)/num_classes, sum(sen)/num_classes, sum(spe)/num_classes, auc
    else:
        return sum(acc)/num_classes, sum(sen)/num_classes, sum(spe)/num_classes


def cemap_cal(y_pred,y_true):
    """
    function to calculate C-MAP(mAP) and E-MAP
    y_true: 0 1
    """
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0,nTest):
        R = np.sum(y_true[i,:])
        for j in range(0,nLabel):            
            if y_true[i,j]==1:
                r = np.sum(y_pred[i,:]>=y_pred[i,j])
                rb = np.sum(y_pred[i,np.nonzero(y_true[i,:])] >= y_pred[i,j])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)

    ap = np.zeros(nLabel)
    for i in range(0,nLabel):
        R = np.sum(y_true[:,i])
        for j in range(0,nTest):
            if y_true[j,i]==1:
                r = np.sum(y_pred[:,i] >= y_pred[j,i])
                rb = np.sum(y_pred[np.nonzero(y_true[:,i]),i] >= y_pred[j,i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap,emap


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class OCTDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.labels = pd.read_csv(label_file, header=0)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_id =  self.labels.iloc[idx, 0]
        img_name = os.path.join(self.image_dir, img_id)
        # time1 = time.time()
        # image = Image.open(img_name).convert('L')
        image = Image.open(img_name).convert('RGB')
        # image = io.imread(img_name)
        # print(time.time() - time1)
        label_ = self.labels.iloc[idx, 2:5].values
        label = np.array(label_).astype('double')
        # if len(image.shape)==2:
            # image = np.expand_dims(image,2)
            # image = np.concatenate((image,image,image),axis=2)
        if self.transform:
            image = self.transform(image)

        return image, label#, img_id


class OCTDataset_CAM(Dataset):
    def __init__(self, image_dir, label_file, transform=None, transform1=None):
        self.labels = pd.read_csv(label_file,header=0)
        self.image_dir = image_dir
        self.transform = transform
        self.transform1 = transform1

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        img_id =  self.labels.iloc[idx,0]
        # print(img_id)
        img_name = os.path.join(self.image_dir, img_id)
        image = cv2.imread(img_name, 0)
        # image = io.imread(img_name)
        # image = Image.open(img_name).convert('RGB')
        raw_image = image
        label = self.labels.iloc[idx,2:5].values
        label = label.astype('double')
        # if len(image.shape)==2:
        #     image = np.expand_dims(image,2)
        #     image = np.concatenate((image,image,image),axis=2)
        # if image.shape[2] == 1:
            # image = np.concatenate((image,image,image),axis=2)
            # print(image.shape)
        if self.transform:
            image = self.transform(image)
        return self.transform1(raw_image), image, label, img_id


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


