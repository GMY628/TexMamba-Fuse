import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
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


data_name = "RealMFF"
img_size = 128
stride = 200

imageA_files = sorted(get_img_file(r"RealMFF/imageA"))
imageB_files = sorted(get_img_file(r"RealMFF/imageB"))
mask_files = sorted(get_img_file(r"RealMFF/RealMFFmask"))
seg_files = sorted(get_img_file(r"RealMFF/Fusion"))

assert len(imageA_files) == len(imageB_files)
h5f = h5py.File(os.path.join('./data',
                             data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),
                'w')
h5_imageA = h5f.create_group('A_patchs')
h5_imageB = h5f.create_group('B_patchs')
h5_mask = h5f.create_group('mask_patchs')
h5_seg = h5f.create_group('seg_patchs')
train_num = 0
for i in tqdm(range(len(imageA_files))):
    I_imageA = imread(imageA_files[i]).astype(
        np.float32).transpose(2, 0, 1)/255.
    I_imageB = imread(imageB_files[i]).astype(
        np.float32).transpose(2, 0, 1)/255.
    I_mask = imread(mask_files[i]).astype(np.float32)[None, :, :]/255.
    I_seg = imread(seg_files[i]).astype(np.float32).transpose(2, 0, 1)/255.

    I_imageA_Patch_Group = Im2Patch(I_imageA, img_size, stride)
    I_imageB_Patch_Group = Im2Patch(I_imageB, img_size, stride)
    I_mask_Patch_Group = Im2Patch(I_mask, img_size, stride)
    I_seg_Patch_Group = Im2Patch(I_seg, img_size, stride)

    for ii in range(I_imageA_Patch_Group.shape[-1]):

        avl_imageA = I_imageA_Patch_Group[:, :, :, ii]
        avl_imageB = I_imageB_Patch_Group[:, :, :, ii]
        avl_mask = I_mask_Patch_Group[:, :, :, ii]
        avl_seg = I_seg_Patch_Group[:, :, :, ii]

        h5_imageA.create_dataset(str(train_num),     data=avl_imageA,
                                 dtype=avl_imageA.dtype,   shape=avl_imageA.shape)
        h5_imageB.create_dataset(str(train_num),    data=avl_imageB,
                                 dtype=avl_imageB.dtype,  shape=avl_imageB.shape)
        h5_mask.create_dataset(str(train_num),    data=avl_mask,
                               dtype=avl_mask.dtype,  shape=avl_mask.shape)
        h5_seg.create_dataset(str(train_num),    data=avl_seg,
                              dtype=avl_seg.dtype,  shape=avl_seg.shape)
        train_num += 1

h5f.close()

with h5py.File(os.path.join('data',
                            data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'), "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
