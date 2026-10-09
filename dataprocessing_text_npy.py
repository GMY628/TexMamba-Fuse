import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
import torch
from skimage.io import imread
from tqdm import tqdm
import numpy as np
import h5py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy', 'txt')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * \
        0.587000 + img[2:3, :, :] * 0.114000
    return y


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
print("blip2")
for param in model.parameters():
    param.requires_grad = False

data_name = "RealMFF639"
img_size = 768
stride = 0

imageA_files = sorted(get_img_file(r"RealMFF639/imageA"))
imageB_files = sorted(get_img_file(r"RealMFF639/imageB"))
mask_files = sorted(get_img_file(r"RealMFF639/RealMFFmask"))
seg_files = sorted(get_img_file(r"RealMFF639/Fusion"))
text_nearunique_files = sorted(get_img_file(r"RealMFF639/text_nearunique"))
text_farunique_files = sorted(get_img_file(r"RealMFF639/text_farunique"))
text_near_far_common_files = sorted(
    get_img_file(r"RealMFF639/text_near_far_common"))

assert len(imageA_files) == len(imageB_files)
h5f = h5py.File(os.path.join('./data',
                             data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+"_text"+'.h5'),
                'w')
h5_imageA = h5f.create_group('A_patchs')
h5_imageB = h5f.create_group('B_patchs')
h5_mask = h5f.create_group('mask_patchs')
h5_seg = h5f.create_group('seg_patchs')
h5_text_nearunique = h5f.create_group('text_nearunique_patchs')
h5_text_farunique = h5f.create_group('text_farunique_patchs')
h5_text_near_far_common = h5f.create_group('text_near_far_common_patchs')
train_num = 0
for i in tqdm(range(len(imageA_files))):
    I_imageA = imread(imageA_files[i]).astype(
        np.float32).transpose(2, 0, 1)/255.
    I_imageB = imread(imageB_files[i]).astype(
        np.float32).transpose(2, 0, 1)/255.
    I_mask = imread(mask_files[i]).astype(np.float32)[None, :, :]/255.
    I_seg = imread(seg_files[i]).astype(np.float32).transpose(2, 0, 1)/255.

    with torch.no_grad(), torch.cuda.amp.autocast():
        with open(text_nearunique_files[i], 'r') as file:
            T_nearfocus = file.read()
            tempnear = txt_processors["eval"](T_nearfocus)
            sample1 = {"text_input": [tempnear]}
            T_nearfocus = model.extract_features(
                sample1, mode="text").text_embeds[0]

            n = 147 - T_nearfocus.shape[0]

            T_nearfocus = F.pad(T_nearfocus, (0, 0, 0, n))

            T_nearfocus = T_nearfocus.cpu().numpy()
        with open(text_farunique_files[i], 'r') as file:
            T_farfocus = file.read()
            tempfar = txt_processors["eval"](T_farfocus)
            sample2 = {"text_input": [tempfar]}
            T_farfocus = model.extract_features(
                sample2, mode="text").text_embeds[0]
            n = 147 - T_farfocus.shape[0]
            T_farfocus = F.pad(T_farfocus, (0, 0, 0, n))
            T_farfocus = T_farfocus.cpu().numpy()
        with open(text_near_far_common_files[i], 'r') as file:
            T_near_far_common = file.read()
            tempcommon = txt_processors["eval"](T_near_far_common)
            sample3 = {"text_input": [tempcommon]}
            T_near_far_common = model.extract_features(
                sample3, mode="text").text_embeds[0]
            n = 147 - T_near_far_common.shape[0]
            T_near_far_common = F.pad(T_near_far_common, (0, 0, 0, n))
            T_near_far_common = T_near_far_common.cpu().numpy()

    for ii in range(1):

        avl_imageA = I_imageA
        avl_imageB = I_imageB
        avl_mask = I_mask
        avl_seg = I_seg
        avl_T_nearfocus = T_nearfocus
        avl_T_farfocus = T_farfocus
        avl_T_near_far_common = T_near_far_common

        h5_imageA.create_dataset(str(train_num),     data=avl_imageA,
                                 dtype=avl_imageA.dtype,   shape=avl_imageA.shape)
        h5_imageB.create_dataset(str(train_num),    data=avl_imageB,
                                 dtype=avl_imageB.dtype,  shape=avl_imageB.shape)
        h5_mask.create_dataset(str(train_num),    data=avl_mask,
                               dtype=avl_mask.dtype,  shape=avl_mask.shape)
        h5_seg.create_dataset(str(train_num),    data=avl_seg,
                              dtype=avl_seg.dtype,  shape=avl_seg.shape)
        h5_text_nearunique.create_dataset(str(train_num),    data=avl_T_nearfocus,
                                          dtype=avl_T_nearfocus.dtype,  shape=avl_T_nearfocus.shape)
        h5_text_farunique.create_dataset(str(train_num),    data=avl_T_farfocus,
                                         dtype=avl_T_farfocus.dtype,  shape=avl_T_farfocus.shape)
        h5_text_near_far_common.create_dataset(str(train_num),    data=avl_T_near_far_common,
                                               dtype=avl_T_near_far_common.dtype,  shape=avl_T_near_far_common.shape)

        train_num += 1

h5f.close()

with h5py.File(os.path.join('data',
                            data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+"_text"+'.h5'), "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
