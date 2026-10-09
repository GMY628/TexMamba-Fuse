import torch.utils
from mambablock_open import Mamba_Encoder, Mamba_Decoder, Mamba_Decoder2, UNet

import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
from torchvision.utils import save_image
import skimage
from skimage import morphology
import torch.nn.functional as f
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


def box_filter(imgSrc, r):

    if imgSrc.ndim == 2:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape[:2])

        imCum = np.cumsum(imgSrc, axis=0)

        imDst[0: r+1] = imCum[r: 2 * r+1]
        imDst[r + 1: h - r] = imCum[2 * r + 1: h] - imCum[0: h - 2 * r - 1]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :],
                                     [r, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        imCum = np.cumsum(imDst, axis=1)

        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - \
            imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r]) - \
            imCum[:, w - 2 * r - 1: w - r - 1]
    else:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape)

        imCum = np.cumsum(imgSrc, axis=0)

        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - \
            imCum[0: h - 2 * r - 1, :]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :],
                                     [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        imCum = np.cumsum(imDst, axis=1)

        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - \
            imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - \
            imCum[:, w - 2 * r - 1: w - r - 1]
    return imDst


def guided_filter(I, p, r, eps=0.1):
    h, w = I.shape[:2]
    if I.ndim == 2:
        N = box_filter(np.ones((h, w)), r)
    else:
        N = box_filter(np.ones((h, w, 1)), r)
    mean_I = box_filter(I, r) / N
    mean_p = box_filter(p, r) / N
    mean_Ip = box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)

    if I.ndim == 2:
        b = mean_p - a * mean_I
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = mean_a * I + mean_b
    else:
        b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
    return q


def fusion_channel_sf(f1, f2, kernel_radius=5):
    """
    Perform channel sf fusion two features
    """
    device = f1.device
    b, c, h, w = f1.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])\
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    f1_r_shift = f.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = f.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    f2_r_shift = f.conv2d(f2, r_shift_kernel, padding=1, groups=c)
    f2_b_shift = f.conv2d(f2, b_shift_kernel, padding=1, groups=c)

    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones(
        (c, 1, kernel_size, kernel_size)).float().cuda(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(f.conv2d(f1_grad, add_kernel,
                      padding=kernel_padding, groups=c), dim=1)
    f2_sf = torch.sum(f.conv2d(f2_grad, add_kernel,
                      padding=kernel_padding, groups=c), dim=1)
    weight_zeros = torch.zeros(f1_sf.shape).cuda(device)
    weight_ones = torch.ones(f1_sf.shape).cuda(device)

    dm_tensor = torch.where(f1_sf > f2_sf, weight_ones,
                            weight_zeros).cuda(device)

    return dm_tensor


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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r"checkpoints\stage1.pth"
for dataset_name in ['lytro']:
    print("\n"*2+"="*80)

    print("The test result of "+dataset_name+' :')

    test_folder = os.path.join('test_img', dataset_name)
    test_out_folder = os.path.join('test_result_decison_map', dataset_name)
    if not os.path.exists(test_out_folder):
        os.makedirs(test_out_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Encoder_common = nn.DataParallel(Mamba_Encoder()).to(device)
    Encoder_diff = nn.DataParallel(Mamba_Encoder(inp_channels=3,)).to(device)

    Decoder_return = nn.DataParallel(Mamba_Decoder(dim=128)).to(device)
    Decoder_mask = nn.DataParallel(UNet()).to(device)

    Encoder_common.load_state_dict(
        torch.load(ckpt_path)['DIDF_Encoder_common'])
    Encoder_diff.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder_diff'])
    Decoder_return.load_state_dict(
        torch.load(ckpt_path)['DIDF_Decoder_return'])
    Decoder_mask.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder_mask'])

    Encoder_common.eval()
    Encoder_diff.eval()
    Decoder_return.eval()
    Decoder_mask.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "blur1")):

            imageA = image_read_cv2(os.path.join(
                test_folder, "blur1", img_name), mode='RGB').transpose(2, 0, 1)[np.newaxis, ...]/255.0
            imageB = image_read_cv2(os.path.join(
                test_folder, "blur2", img_name), mode='RGB').transpose(2, 0, 1)[np.newaxis, ...]/255.0
            name = img_name[:-4]
            imageA, imageB = torch.FloatTensor(
                imageA), torch.FloatTensor(imageB)
            imageA, imageB = imageA.to(device), imageB.to(device)
            print(name)
            feature_common = Encoder_common(torch.cat((imageA, imageB), dim=1))

            feature_image1 = Encoder_diff(imageA)
            feature_image2 = Encoder_diff(imageB)

            image1_hat = Decoder_return(
                torch.cat((feature_common, feature_image1), dim=1))
            image2_hat = Decoder_return(
                torch.cat((feature_common, feature_image2), dim=1))
            mask_hat = Decoder_mask(
                torch.cat((feature_image1, feature_image2), dim=1))

            mask_sf = fusion_channel_sf(feature_image1, feature_image2)

            mask_hat = mask_hat > 0.5
            mask_hat = mask_hat.to(torch.float)

            mask_sf = mask_sf > 0.5
            mask_sf = mask_sf.to(torch.float)
            mask_sf = mask_sf.unsqueeze(1)

            dm = remove_small_blobs(mask_hat).squeeze(
            ).cpu().detach().numpy().astype(np.int_)
            dm = np.expand_dims(dm, axis=2)

            imageA = imageA.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            imageB = imageB.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            temp_fused = imageA * dm + imageB * (1 - dm)

            dm = torch.Tensor(dm).permute(2, 0, 1).unsqueeze(0).to(device)

            temp_fused = torch.Tensor(temp_fused.transpose(2, 0, 1)).to(device)

            mask_sf = remove_small_blobs(mask_sf)

            data_common = torch.mean(feature_common, dim=1, keepdim=True)
            data_1 = torch.mean(feature_image1, dim=1, keepdim=True)
            data_2 = torch.mean(feature_image2, dim=1, keepdim=True)

            save_image(dm, test_out_folder+'/'+name+'_mask.png')
