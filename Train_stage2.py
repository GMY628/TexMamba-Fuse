# -*- coding: utf-8 -*-

import random
from skimage import morphology
import skimage
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import save_image
import kornia
from torch.nn import functional as F
from utils.loss import Fusionloss
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import datetime
import time
import sys
from utils.Logger import Logger1
from utils.loss import *
from utils.dataset import H5Dataset, H5Dataset_withtext
from mambablock_open import Mamba_Encoder, UNet
from mambablock_open import Mamba_Decoder_textguide as Mamba_Decoder
from tensorboardX import SummaryWriter
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed_value = 3407
os.environ['PYTHONHASHSEED'] = str(seed_value)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_str = 'mambafuse_stage2_withtext'


num_epochs = 101
epoch_gap = 101

lr = 1e-4
weight_decay = 0
batch_size = 1
print("batchsize", batch_size)
print("num_epochs", num_epochs)


coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 2.
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


def remove_small_blobs(mask):
    dm = mask.squeeze().cpu().detach().numpy().astype(np.int_)
    _, _, h, w = mask.shape

    se = skimage.morphology.disk(1)
    dm = skimage.morphology.binary_opening(dm, se)
    dm = morphology.remove_small_holes(dm == 0, 0.01 * h * w)
    dm = np.where(dm, 0, 1)
    dm = skimage.morphology.binary_closing(dm, se)
    dm = morphology.remove_small_holes(dm == 1, 0.01 * h * w)
    dm = np.where(dm, 1, 0)
    dm = torch.Tensor(dm).unsqueeze(0).unsqueeze(0).to(device)

    return dm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder_common = nn.DataParallel(Mamba_Encoder()).to(device)
DIDF_Encoder_diff = nn.DataParallel(Mamba_Encoder(inp_channels=3,)).to(device)
Decoder_mask = nn.DataParallel(UNet()).to(device)
Mamba_decoder = nn.DataParallel(Mamba_Decoder()).to(device)


optimizer1 = torch.optim.Adam(
    Mamba_decoder.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer1, step_size=optim_step, gamma=optim_gamma)


MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
loss_ortho = OrthoLoss()

smoothl1loss = nn.SmoothL1Loss()
criterion = LpLssimLossweight().to(device)
loss = Fusionloss(coeff_grad=50, device=device)


ckpt_path = r"checkpoints/stage1.pth"

datasetname = "RealMFF639_imgsize_768_stride_0_text.h5"
trainloader = DataLoader(H5Dataset(r"data/RealMFF_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

trainloader_text = DataLoader(H5Dataset_withtext(r"data/RealMFF639_imgsize_768_stride_0_text.h5"),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

loader = {'train': trainloader, 'train_text': trainloader_text}
print(datasetname)
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

logname = 'No10'
if os.path.exists('logs_withtext/' + logname) == False:
    os.mkdir('logs_withtext/' + logname)
logger = Logger1(rootpath='logs_withtext/' + logname, timestamp=False)
params = {
    'epoch': num_epochs,
    'lr': lr,
    'batch_size': batch_size,
    'feature_model': ckpt_path,
    'dataSetname': datasetname,
}
logger.save_param(params)


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
prev_time = time.time()


DIDF_Encoder_common.load_state_dict(
    torch.load(ckpt_path)['DIDF_Encoder_common'])
DIDF_Encoder_diff.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder_diff'])
Decoder_mask.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder_mask'])


DIDF_Encoder_common.eval()
DIDF_Encoder_diff.eval()
Decoder_mask.eval()

for param in DIDF_Encoder_common.parameters():
    param.requires_grad = False

for param in DIDF_Encoder_diff.parameters():
    param.requires_grad = False

for param in Decoder_mask.parameters():
    param.requires_grad = False


for epoch in range(num_epochs):
    ''' train '''
    for i, (image1, image2, _, seg, text_nearunique, text_farunique, text_near_far_common) in enumerate(loader['train_text']):
        image1, image2, _, seg, text_nearunique, text_farunique, text_near_far_common = image1.cuda(), image2.cuda(
        ), _.cuda(), seg.cuda(), text_nearunique.cuda(), text_farunique.cuda(), text_near_far_common.cuda()

        Mamba_decoder.train()

        Mamba_decoder.zero_grad()

        optimizer1.zero_grad()

        if epoch <= epoch_gap:
            feature_common = DIDF_Encoder_common(
                torch.cat((image1, image2), dim=1))

            feature_image1 = DIDF_Encoder_diff(image1)
            feature_image2 = DIDF_Encoder_diff(image2)

            fuse_hat = Mamba_decoder(feature_image1, feature_image2, feature_common,
                                     text_nearunique, text_farunique, text_near_far_common)

            ssim1 = Loss_ssim(image1, fuse_hat1)
            ssim2 = Loss_ssim(image2, fuse_hat2)

            transform = transforms.Grayscale(num_output_channels=1)
            image1 = transform(image1)
            image2 = transform(image2)
            fuse_hat1 = transform(fuse_hat1)
            fuse_hat2 = transform(fuse_hat2)

            loss_in_grad1, loss_in1, loss_grad1 = loss(
                image1, image1, fuse_hat1)
            loss_in_grad2, loss_in2, loss_grad2 = loss(
                image2, image2, fuse_hat2)

            loss_ssim = ssim1 + ssim2
            loss_in_grad = loss_in_grad1 + loss_in_grad2

            lossALL = loss_in_grad + 100 * loss_ssim

            lossALL.backward()
            nn.utils.clip_grad_norm_(
                Mamba_decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()

        batches_done = epoch * len(loader['train_text']) + i
        batches_left = num_epochs * len(loader['train_text']) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(

            "\r[Epoch %d/%d] [Batch %d/%d] [lossALL: %f] [loss_in_grad: %f] [loss_ssim: %f]  ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train_text']),
                lossALL.item(),

                loss_in_grad.item(),
                loss_ssim.item(),

                time_left,
            )
        )

    scheduler1.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6

    if epoch % 5 == 0:
        checkpoint = {
            'DIDF_Encoder_common': DIDF_Encoder_common.state_dict(),
            'DIDF_Encoder_diff': DIDF_Encoder_diff.state_dict(),
            'Mamba_decoder': Mamba_decoder.state_dict(),
        }

        if os.path.exists('models_withtext/' + logname) == False:
            os.mkdir('models_withtext/' + logname)
        torch.save(checkpoint, os.path.join("models_withtext/" + logname + "/" +
                   "stage2_singlebranch_withtext_epoch_"+str(epoch)+"_"+timestamp+'.pth'))
