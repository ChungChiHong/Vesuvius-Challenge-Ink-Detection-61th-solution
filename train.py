# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:32:43 2023

@author: CH
"""
from torch.cuda.amp import autocast, GradScaler

import os
import gc
import math
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import warnings
import torch.nn.functional as F
from einops import rearrange
warnings.filterwarnings("ignore")


model_list = os.listdir("./model/")
model_id = 0
for model_name in model_list:
    if int(model_name.split("_")[0])>model_id:
        model_id=int(model_name.split("_")[0].split(".")[0])
    
model_id += 1
print(f"model id:{model_id}")

class CFG:
    # ============== model cfg =============
    model_name = 'Unet'
    backbone = "tf_efficientnet_b7"

    load = 0 
    start_epoch = 0
    
    drop_rate = 0.5
    drop_path_rate = 0.5

    mosaic = 1
    
    mid = 32
    start = 20 
    end =  40 
    in_chans =  end-start# 65
    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 4
    
    train_batch_size = 8
    valid_batch_size = train_batch_size * 1
    use_amp = True
    
    # ============== label_smoothing =============
    apply_label_smoothing = 1
    pos_label = 1
    neg_label = 0.2

    # ============== fold =============
    valid_id = 1

    # ============== fixed =============
    train = 1
    
    # adamW warmupあり
    epochs = 10 # 30
    warmup_epoch = 1
    
    warmup_factor = 10
    lr = 3e-4 /warmup_factor 
    min_lr = 1e-5 
    weight_decay = 1e-6
    max_grad_norm = 1000

    num_workers = 0

    seed = 42
    # ============== augmentation =============
       
    train_aug_list = [
        
        #A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.RandomGridShuffle(grid=(3,3),p=0.5),

        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),

    ]

    valid_aug_list = [
        #A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    # ============== comp exp name =============

    comp_dir_path = './'
    comp_folder_name = 'img'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    exp_name = model_id
    
    # ============== set dataset path =============

    outputs_path = './'
    model_dir = outputs_path + 'model/'
    log_dir = outputs_path + 'logs/'
    mask_dir = outputs_path + 'predict_mask/'
    log_path = log_dir + f'{exp_name}.txt'
    
    # ============== pred target =============
    target_size = 1

# In[ ]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


# In[ ]:

def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.log_dir, cfg.mask_dir]:
        os.makedirs(dir, exist_ok=True)

# In[ ]:

def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)

# In[ ]:

with open(CFG.log_path, 'w') as f:
    for attr in dir(CFG):
        if not callable(getattr(CFG, attr)) and not attr.startswith("__"):
            f.write('{} = {}\n'.format(attr, getattr(CFG, attr)))
            
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Logger = init_logger(log_file=CFG.log_path)


# In[ ]:


def read_image_mask(fragment_id):

    images = []
    start = CFG.start
    end = CFG.end
    
    idxs = range(start, end)
    bar = tqdm(idxs,total=len(idxs))
    
    for i in bar:
        if fragment_id==2:
            image = cv2.imread(CFG.comp_dataset_path + f"train/2/surface_volume/{i:02}.tif", 0)
            image = image[:int(image.shape[0]/2),:]
        elif fragment_id==4:
            image = cv2.imread(CFG.comp_dataset_path + f"train/2/surface_volume/{i:02}.tif", 0)
            image = image[int(image.shape[0]/2)+1:,:]
        else:
            image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)
         
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

        bar.set_postfix(Fragment_id=fragment_id)
        
    bar.close() 
    
    images = np.stack(images, axis=2)

    if fragment_id==2:
        mask = cv2.imread(CFG.comp_dataset_path + "train/2/inklabels.png", 0)
        mask = mask[:int(mask.shape[0]/2),:]
    elif fragment_id==4:
        mask = cv2.imread(CFG.comp_dataset_path + "train/2/inklabels.png", 0)
        mask = mask[int(mask.shape[0]/2)+1:,:]
    else:
        mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0
    
    return images, mask


# In[ ]:


def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    on_papyrus = []
    positive_mask = []
    
    for fragment_id in range(1, 5):

        image, mask = read_image_mask(fragment_id)
        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
        
        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size

                if fragment_id == CFG.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])
                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    if image[y1:y2, x1:x2].max()!=0:
                        train_images.append(image[y1:y2, x1:x2])
                        train_masks.append(mask[y1:y2, x1:x2, None])
                        
                        on_papyrus.append(True)
                            
                        if mask[y1:y2, x1:x2, None].max()==0:
                            positive_mask.append(False)
                        else:
                            positive_mask.append(True)
                    
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys, on_papyrus, positive_mask


# In[ ]:

train_images, train_masks, valid_images, valid_masks, valid_xyxys, on_papyrus, positive_mask = get_train_valid_dataset()

# In[ ]:

valid_xyxys = np.stack(valid_xyxys)

# In[ ]:

def mosaic4(idx, images_list, labels_list, on_papyrus_True, positive_mask_True):

    images = []
    labels = []
    images.append(images_list[idx])
    labels.append(labels_list[idx])
    
    contain_positive_mask = 0
    
    for i in range(3):

        if contain_positive_mask==0:
            
            n = np.random.choice(positive_mask_True)     
            images.append(images_list[n])
            labels.append(labels_list[n])
            contain_positive_mask=1
            
        else:
            n = np.random.choice(on_papyrus_True)     
            images.append(images_list[n])
            labels.append(labels_list[n])
        
    top_row = cv2.hconcat([images[0], images[1]])
    bottom_row = cv2.hconcat([images[2], images[3]])
    image = cv2.vconcat([top_row, bottom_row])
    
    top_row = cv2.hconcat([labels[0], labels[1]])
    bottom_row = cv2.hconcat([labels[2], labels[3]])
    label = cv2.vconcat([top_row, bottom_row])
     
    left = np.random.randint(0, image.shape[1] - CFG.size + 1)
    top = np.random.randint(0, image.shape[0] - CFG.size + 1)
    cropped_image = image[top:top+CFG.size, left:left+CFG.size]
    cropped_mask = label[top:top+CFG.size, left:left+CFG.size]
    cropped_mask = np.expand_dims(cropped_mask, axis=-1)
    
    return cropped_image,cropped_mask
    
def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None, train=False, on_papyrus=[], positive_mask=[]):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.train = train
        self.on_papyrus_True = np.where(np.array(on_papyrus) == True)[0] # on_papyrus中為True的位置
        self.positive_mask_True = np.where(np.array(positive_mask) == True)[0]
        
        
        
    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.train and self.cfg.mosaic:
            image, label = mosaic4(idx, self.images, self.labels, self.on_papyrus_True, self.positive_mask_True)

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image, label


# In[ ]:

train_dataset = CustomDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG), train=True, on_papyrus=on_papyrus, positive_mask=positive_mask)
valid_dataset = CustomDataset(
    valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

gc.collect()

# In[ ]:

import timm
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock

class SmpUnetDecoder(UnetDecoder):
	def __init__(self, **kwargs):
		super(SmpUnetDecoder, self).__init__(
			**kwargs)

	def forward(self, encoder):
		feature = encoder[::-1]  # reverse channels to start from head of encoder
		head = feature[0]
		skip = feature[1:] + [None]
		d = self.center(head)

		decoder = []
		for i, decoder_block in enumerate(self.blocks):
			# print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
			# print(decoder_block.conv1[0])
			# print('')
			s = skip[i]
			d = decoder_block(d, s)
			decoder.append(d)

		last  = d
		return last, decoder

class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()

        self.encoder = timm.create_model(
            cfg.backbone,
            in_chans=4,
            features_only=True,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            pretrained=True
        )
        g = self.encoder(torch.rand(1, 4, cfg.size, cfg.size))
        
        encoder_dim =   [_.shape[1] for _ in g]

        self.decoder = SmpUnetDecoder(
        	encoder_channels= [0] + encoder_dim,
			decoder_channels=[256, 128, 64, 32, 16],
			n_blocks=5,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)
        
        #-- pool attention weight
        self.weight = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
			) for dim in encoder_dim
		])
        
        self.logit = nn.Conv2d(16,1,kernel_size=1)

        self.encoder2 = timm.create_model(
            "tf_efficientnet_b0",
            in_chans=1,
            features_only=True,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            pretrained=True
        )
        g = self.encoder2(torch.rand(1, 1, cfg.size, cfg.size))
        
        encoder_dim2 =   [_.shape[1] for _ in g]

        self.decoder2 = SmpUnetDecoder(
        	encoder_channels= [0] + encoder_dim2,
			decoder_channels=[256, 128, 64, 32, 16],
			n_blocks=5,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)
        self.logit2 = nn.Conv2d(16,1,kernel_size=1)
        
	
    def forward(self,input):
        v = input
        B,C,H,W = v.shape
        vv = [
            	v[:,i:i+4] for i in range(0,20-4,2)
        	]
        K = len(vv)
        x = torch.cat(vv,0)
        
        encoder = self.encoder(x)
        for i in range(len(encoder)):
            e = encoder[i]
            f = self.weight[i](e)
            _, c, h, w = f.shape
            f = rearrange(f, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #f.reshape(B, K, c, h, w)
            e = rearrange(e, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #e.reshape(B, K, c, h, w)
            w = F.softmax(f, 1)
            e = (w * e).sum(1)
            encoder[i] = e

        last, decoder = self.decoder(encoder)
        logit = self.logit(last)
        
        # ----------------
        encoder2 = self.encoder2(logit)
        last2, decoder2 = self.decoder2(encoder2)
        logit2 = self.logit2(last2)
        
        return logit,logit2
    
def build_model(cfg): 
   
    model = CustomModel(cfg)
    return model

# In[ ]:

import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs-cfg.warmup_epoch-1, eta_min=CFG.min_lr)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=cfg.warmup_factor, total_epoch=cfg.warmup_epoch, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, epoch):
    scheduler.step(epoch)

# In[ ]:

model = build_model(CFG)
if CFG.load:
    state = torch.load(CFG.model_dir + f'{CFG.load}_best_cv.pth')
    model.load_state_dict(state)
model.to(device)

optimizer = AdamW(model.parameters(), lr=CFG.lr)
scheduler = get_scheduler(CFG, optimizer)

# In[ ]:

    
def label_smoothing(label,pos_label=CFG.pos_label,neg_label=CFG.neg_label):
    label = label*pos_label
    label = torch.where(label < neg_label, torch.tensor(neg_label).to(device), label)
    
    return label
       
    
# In[ ]:

DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
FocalLoss = smp.losses.FocalLoss(mode='binary')


alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(
    mode='binary', log_loss=False, alpha=alpha, beta=beta)


def criterion(y_pred, y_true):
    
    if CFG.apply_label_smoothing:
       y_true = label_smoothing(y_true, pos_label=CFG.pos_label, neg_label=CFG.neg_label)

    return  BCELoss(y_pred, y_true)*0.5 + DiceLoss(y_pred, y_true)*0.5


# In[ ]:

def train_fn(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, labels) in bar:

        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()

        with autocast(CFG.use_amp):
            y_preds,y_preds2 = model(images)
            loss = criterion(y_preds, labels)
            loss2 = criterion(y_preds2, labels)
            loss = loss*0.5 + loss2*0.5
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        
        bar.set_postfix(Epoch=epoch, 
                        Lr=optimizer.param_groups[0]['lr'], 
                        Train_Loss=losses.avg)
    bar.close()
    return losses.avg

def valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, epoch):
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (images, labels) in bar:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds,y_preds2 = model(images)
            loss = criterion(y_preds, labels)
            loss2 = criterion(y_preds2, labels)
            loss = loss*0.5 + loss2*0.5
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        start_idx = step*CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))
            
        bar.set_postfix(Epoch=epoch,
                        Valid_Loss=losses.avg)
    bar.close()

    mask_pred /= mask_count
    
    best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

    score = best_dice
    
    return losses.avg, best_th, score, mask_pred

# In[ ]:

from sklearn.metrics import fbeta_score

def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(5, 95+1, 5)) / 100:
        
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)

        if dice > best_dice:
            best_dice = dice
            best_th = th
    
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th

# In[ ]:

fragment_id = CFG.valid_id

if fragment_id==2:
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/2/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt[:int(valid_mask_gt.shape[0]/2),:]
elif fragment_id==4:
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/2/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt[int(valid_mask_gt.shape[0]/2)+1:,:] 
else:
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
valid_mask_gt = valid_mask_gt / 255
pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)


# In[ ]:

best_score = -1
best_loss = np.inf

epoch = 1
if CFG.load:
    epoch = CFG.start_epoch
    
while epoch<=CFG.epochs:
    
        start_time = time.time()
        
        Logger.info("-"*80)
        # train
        #avg_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch)
        # eval
        avg_val_loss, best_th, score, mask_pred = valid_fn(
            valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, epoch)
    
        scheduler_step(scheduler, epoch)
        elapsed = time.time() - start_time
        
        Logger.info(
            f'Epoch {epoch} -  Threshold: {best_th}   Score: {score:.4f}  loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        if score > best_score:
            
            best_score = score
            Logger.info(
                f'*************** - Save Best Score: {best_score:.4f} Model')
            
            torch.save( model.state_dict(),
                        CFG.model_dir + f'{CFG.exp_name}_best_cv.pth')
            
            np.save(f'{CFG.mask_dir}/{CFG.exp_name}_mask.npy', mask_pred)

        epoch+=1
                  
print(f"model id:{model_id}")            
            
del Logger

# In[ ]:

for i in ["cv"]:
    mask_pred = np.load(f'predict_mask/{CFG.exp_name}_best_{i}.npy')
    
    best_dice, best_th  = calc_fbeta(valid_mask_gt, mask_pred)
       
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].imshow(valid_mask_gt)
    axes[1].imshow(mask_pred)
    axes[2].imshow((mask_pred>=best_th).astype(int))
       
    plt.hist(mask_pred.flatten(), bins=20)

