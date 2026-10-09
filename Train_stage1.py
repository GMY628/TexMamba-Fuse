# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""

import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.models as models
import random
import kornia
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import datetime
import time
import sys
from utils.dataset import H5Dataset

from mambablock_open import Mamba_Encoder, Mamba_Decoder, Mamba_Decoder2, UNet
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


seed_value = 3407


os.environ["PYTHONHASHSEED"] = str(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_str = "mambafuse_stage1"


num_epochs = 101
epoch_gap = 101

lr = 1e-4
weight_decay = 0
batch_size = 1
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]

coeff_mse_loss_VF = 1.0
coeff_mse_loss_IF = 1.0
coeff_decomp = 2.0
coeff_tv = 2.5

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


class OrthoLoss(nn.Module):

    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2 = input1
        input2_l2 = input2

        ortho_loss = 0
        dim = input1.shape[1]
        for i in range(input1.shape[0]):
            ortho_loss += torch.mean(
                ((input1_l2[i: i + 1, :].mm(input2_l2[i: i + 1, :].t())).pow(2)) / dim
            )

        ortho_loss = ortho_loss / input1.shape[0]

        return ortho_loss


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


"""
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
"""


def criterion(inputs, target):
    losses1 = [
        F.binary_cross_entropy_with_logits(inputs[i], target)
        for i in range(len(inputs))
    ]
    losses2 = [MSELoss(inputs[i], target) for i in range(len(inputs))]

    total_loss = 0.5 * (sum(losses1) / len(inputs)) + \
        0.5 * (sum(losses2) / len(inputs))

    return total_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
DIDF_Encoder_common = nn.DataParallel(Mamba_Encoder()).to(device)
DIDF_Encoder_diff = nn.DataParallel(
    Mamba_Encoder(
        inp_channels=3,
    )
).to(device)
DIDF_Decoder_return = nn.DataParallel(Mamba_Decoder(dim=128)).to(device)
DIDF_Decoder_mask = nn.DataParallel(UNet(dim=128)).to(device)


""" BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device) """


optimizer1 = torch.optim.Adam(
    DIDF_Encoder_common.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer2 = torch.optim.Adam(
    DIDF_Encoder_diff.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer3 = torch.optim.Adam(
    DIDF_Decoder_return.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer4 = torch.optim.Adam(
    DIDF_Decoder_mask.parameters(), lr=lr, weight_decay=weight_decay
)


""" optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay) """

scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer1, step_size=optim_step, gamma=optim_gamma
)
scheduler2 = torch.optim.lr_scheduler.StepLR(
    optimizer2, step_size=optim_step, gamma=optim_gamma
)
scheduler3 = torch.optim.lr_scheduler.StepLR(
    optimizer3, step_size=optim_step, gamma=optim_gamma
)
scheduler4 = torch.optim.lr_scheduler.StepLR(
    optimizer4, step_size=optim_step, gamma=optim_gamma
)

""" scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma) """

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction="mean")
loss_ortho = OrthoLoss()

smoothl1loss = nn.SmoothL1Loss()
bceloss = nn.BCELoss()

dataset_name = "duts_patches_blurred_boundarymask"
trainloader = DataLoader(
    H5Dataset(r"data/duts_blurred_boundarymask_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
print(dataset_name)
loader = {
    "train": trainloader,
}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

"""
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
"""


step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    """train"""
    for i, (image1, image2, mask, seg) in enumerate(loader["train"]):
        image1, image2, mask, seg = (
            image1.cuda(),
            image2.cuda(),
            mask.cuda(),
            seg.cuda(),
        )

        DIDF_Encoder_common.train()
        DIDF_Encoder_diff.train()
        DIDF_Decoder_return.train()
        DIDF_Decoder_mask.train()

        DIDF_Encoder_common.zero_grad()
        DIDF_Encoder_diff.zero_grad()
        DIDF_Decoder_return.zero_grad()
        DIDF_Decoder_mask.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch <= epoch_gap:

            feature_common = DIDF_Encoder_common(
                torch.cat((image1, image2), dim=1))

            feature_image1 = DIDF_Encoder_diff(image1)
            feature_image2 = DIDF_Encoder_diff(image2)

            image1_hat = DIDF_Decoder_return(
                torch.cat((feature_common, feature_image1), dim=1)
            )
            image2_hat = DIDF_Decoder_return(
                torch.cat((feature_common, feature_image2), dim=1)
            )
            mask_hat = DIDF_Decoder_mask(
                torch.cat((feature_image1, feature_image2), dim=1)
            )

            target_ortho1 = 0.5 * loss_ortho(feature_common, feature_image1)
            target_ortho2 = 0.5 * loss_ortho(feature_common, feature_image2)
            target_ortho3 = 1 * loss_ortho(
                gram_matrix(feature_common), gram_matrix(feature_image1)
            )
            target_ortho4 = 1 * loss_ortho(
                gram_matrix(feature_common), gram_matrix(feature_image2)
            )

            sumortholoss = (
                target_ortho1 + target_ortho2 + target_ortho3 + target_ortho4
            )

            maskloss = 0.5 * MSELoss(mask_hat, mask[:, :1, :, :]) + 0.75 * bceloss(
                mask_hat, mask[:, :1, :, :]
            )

            mse_loss_1 = 5 * Loss_ssim(image1, image1_hat) + \
                MSELoss(image1, image1_hat)
            mse_loss_2 = 5 * Loss_ssim(image2, image2_hat) + \
                MSELoss(image2, image2_hat)

            Gradient_loss = L1Loss(
                kornia.filters.SpatialGradient()(image1),
                kornia.filters.SpatialGradient()(image1_hat),
            )

            Gradient_loss2 = L1Loss(
                kornia.filters.SpatialGradient()(image2),
                kornia.filters.SpatialGradient()(image2_hat),
            )

            loss = (
                coeff_mse_loss_VF * mse_loss_1
                + coeff_mse_loss_IF * mse_loss_2
                + coeff_tv * Gradient_loss
                + coeff_tv * Gradient_loss2
                + sumortholoss
                + 10 * maskloss
            )

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_common.parameters(),
                max_norm=clip_grad_norm_value,
                norm_type=2,
            )
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_diff.parameters(),
                max_norm=clip_grad_norm_value,
                norm_type=2,
            )
            nn.utils.clip_grad_norm_(
                DIDF_Decoder_return.parameters(),
                max_norm=clip_grad_norm_value,
                norm_type=2,
            )
            nn.utils.clip_grad_norm_(
                DIDF_Decoder_mask.parameters(),
                max_norm=clip_grad_norm_value,
                norm_type=2,
            )

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        batches_done = epoch * len(loader["train"]) + i
        batches_left = num_epochs * len(loader["train"]) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader["train"]),
                loss.item(),
                time_left,
            )
        )

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()

    """ if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step() """

    if optimizer1.param_groups[0]["lr"] <= 1e-6:
        optimizer1.param_groups[0]["lr"] = 1e-6
    if optimizer2.param_groups[0]["lr"] <= 1e-6:
        optimizer2.param_groups[0]["lr"] = 1e-6
    if optimizer3.param_groups[0]["lr"] <= 1e-6:
        optimizer3.param_groups[0]["lr"] = 1e-6
    if optimizer4.param_groups[0]["lr"] <= 1e-6:
        optimizer4.param_groups[0]["lr"] = 1e-6

    if epoch % 1 == 0:
        checkpoint = {
            "DIDF_Encoder_common": DIDF_Encoder_common.state_dict(),
            "DIDF_Encoder_diff": DIDF_Encoder_diff.state_dict(),
            "DIDF_Decoder_return": DIDF_Decoder_return.state_dict(),
            "DIDF_Decoder_mask": DIDF_Decoder_mask.state_dict(),
        }
        """ checkpoint = {
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        } """
        torch.save(
            checkpoint,
            os.path.join(
                "newfeaturemodels/"
                + dataset_name
                + "No1_stage1_singlebranch_poch_"
                + str(epoch)
                + "_"
                + timestamp
                + ".pth"
            ),
        )
